import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math

from .build import ENTROPY_MODEL_REGISTRY
LOG2 = math.log(2.)

class CDFLayer(nn.Module):
    """
    layer of CDF estimator
    """
    def __init__(self, channels, in_dim, out_dim, init_scale, act=True):
        super(CDFLayer, self).__init__()
        self.act = act
        weight_init_val = np.log(np.expm1(1 / init_scale / out_dim))
        self.weight = nn.Parameter(torch.nn.init.constant_(
            torch.empty(1, channels, out_dim, in_dim), weight_init_val))
        
        self.bias = nn.Parameter(torch.nn.init.uniform_(
            torch.empty(1, channels, out_dim, 1), -0.5, 0.5))
        
        if act:
            self.factor = nn.Parameter(torch.nn.init.zeros_(
                torch.empty(1, channels, out_dim , 1)))

    def forward(self, x):
        """
        x: shape (N, C, D, 1)
        return output of shape (N, C, D', 1)
        
        NOTE: feature for each channel is of dimension D
        """
        x = torch.matmul(F.softplus(self.weight), x) + self.bias
        if self.act:
            return x + torch.tanh(x) * torch.tanh(self.factor)
        
        return x


class CDFEstimator(nn.Module):
    """
    learn cdf of the given inputs
    cdf for each channel is learned independently
    """
    def __init__(self, channels, dims=[3,3,3], init_scale=10.):
        super().__init__()
        num_layers = len(dims) + 1
        scale = init_scale ** (1 / num_layers)
        dims.insert(0,1)
        dims.append(1)
        
        layers = []
        for i in range(num_layers):
            act = i < (num_layers-1)
            layers.append(CDFLayer(channels, dims[i], dims[i+1], scale, act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: shape (N,C,*)
        
        return cdf of the same shape
        """
        N, C, *spatial_dims = x.size()
        dim_indices = list(range(len(spatial_dims)+2))
        dim_indices.pop(1)
        dim_indices.append(1)
        x = x.permute(*dim_indices).reshape(-1,C,1,1)
        x = self.layers(x)
        x = x.view(N, *spatial_dims, C).permute(*dim_indices)
        return x 


class BaseEntropyModel(nn.Module):
    
    def forward(self, x):
        """
        x: shape (N,C,*)
        """
        mode = "noise" if self.training else "quantize"
        quantized_x = self._quantize(x, mode)
        likelihood = self._prob_mass(quantized_x)
        return quantized_x, likelihood
    
    def _prob_mass(self, x):
        """
        get prob mass for the quantized value
        
        args:
            x: values to get theirs probability shape (N,C,*)
            
        return probability of the same shape
        """
        raise NotImplementedError("to be inherited")
    
    def _quantize(self, x, mode):
        """
        quantize representation to discrete value
        
        For training, uniform(-.5,.5) noise is added to x.
        In inference, return round(x+0.5 - median)
        args
            --mode(str): either "noise" of or "symbol"
            --x: tensor of shape #########
        """
        raise NotImplementedError()
    
    def _dequantize(self,x):
        "inverse of _quantize method, used for inference only"
        raise NotImplementedError()
    
    def compress(self, x):
        """
        compress inputs to binary representation as string
        x : shape (N,C,*)
        """
        raise NotImplementedError()
    
    def decompress(self, x):
        """
        decompress from binary representation
        x : string
        """
        raise NotImplementedError()
    
    def _ce_loss(self, probs):
        """
        cross entropy loss H(p,q) 
        where p is the actual distribution from which inputs is sampled
            q is the estimator of p
        
        
        In end-to-end training, the loss function simutaneously learning q as well as minimize
        entropy of p  (H(p,q) = KL(p || q) + H(p))
        
        NOTE: H is computed as mean per pixel in base 2 in order to make loss scale independent
            from batch size and image size 
        """
        
        # print(probs.min(), probs.max(), torch.isnan(torch.log(probs + 1e-5)).any(), probs.size())
        N,C, *_ = probs.size()
        return (torch.clamp(-1.0 * torch.log(probs + 1e-10) / LOG2, 0, 50)).mean() * C


@ENTROPY_MODEL_REGISTRY.register()
class EntropyModel(BaseEntropyModel):
    def __init__(self, in_channels, cfg):
        super().__init__()
        
        # cdf estimator
        dims                = cfg.MODEL.ENTROPY_MODEL.DIMS
        init_scale          = cfg.MODEL.ENTROPY_MODEL.INIT_SCALE
        self._cdf_estimator = CDFEstimator(in_channels, dims, init_scale)
        
        self.bin = cfg.MODEL.ENTROPY_MODEL.BIN
        
        # init lower, upper, median quantiles
        # self._quantiles = nn.Parameter()
        # self.register_parameter("quantiles", self._quantiles)
    
    def forward(self, x):
        """
        x: shape (N,C,*)
        """
        mode = "noise" if self.training else "quantize"
        quantized_x = self._quantize(x, mode)
        probs = self._prob_mass(quantized_x)
        # aux_loss = self._aux_loss()
        # if self.training:
        ce_loss = self._ce_loss(probs)
        return quantized_x, probs, ce_loss, #aux_loss
        # return quantized_x, probs
    

    # def _aux_loss(self, ):
    #     """
    #     loss for estimating upper and lower quantiles as well as median
    #     """
    #     pass
    
    def _quantize(self, x, mode):
        """
        quantize representation to discrete value
        
        "noise" mode for training, uniform(-.5,.5) noise is added to x.
        "quantize" mode for evaluating perfomance only, return round(x+0.5).
        "symbol" mode to actually encode given values, return round(x+0.5-mean)
        
        args
            --mode(str): either "noise" of or "quantize" or "symbol"
            --x: tensor of shape #########
        """
        half = self.bin / 2
        if mode == "noise":
            noise = torch.rand_like(x) - half
            noise.detach_()
            return x + noise
        elif mode == "quantize":
            return torch.round(x)
        elif mode == "symbol":
            raise NotImplementedError()
    
    def _dequantize(self,x):
        raise NotImplementedError("to be added")
    
    def compress(self, x):
        """
        to be added
        """
        raise NotImplementedError()
    
    def decompress(self, x):
        """
        to be added
        """
        raise NotImplementedError()
    
    def _logit_cumulative(self, x):
        """
        x: shape (N,C,*)
        """
        return self._cdf_estimator(x)
    
    def _prob_mass(self, x):
        """
        x shape (N,C,*)
        return probability of the same shape
        """
        half = self.bin/2
        lower = self._logit_cumulative(x-half)
        upper = self._logit_cumulative(x+half)
        sign = -torch.sign(lower+upper).detach()
        prob = sign*(torch.sigmoid(upper*sign) - torch.sigmoid(lower*sign))
        return prob
    
    
class SymmetricConditionalModel(BaseEntropyModel):
    """
    base class for symmetric distribution conditioned on scale parameter
    """
    def __init__(self, cfg):
        super().__init__()
        self.bin = cfg.MODEL.ENTROPY_MODEL.BIN

    def forward(self, x, scale, mean=0):
        """
        x: shape (N,C,*)
        """
        # set scale and mean as attribute to use in other funtion
        self._scale = scale
        self._mean = mean
        
        mode = "noise" if self.training else "quantize"
        quantized_x = self._quantize(x, mode)
        likelihood = self._prob_mass(quantized_x)
        
        # remove reference to scale and mean
        self._mean = None
        self._scale = None
        return quantized_x, likelihood
    
    def _standardized_cumulative(self, x):
        """
        get cumulative density with scale (std) of 1.
        
        args:
            x: tensor of any shape
        return 
            corresponding cumulative density of the same shape
        """
        raise NotImplementedError()

    def _standardized_quantile(self, x):
        """
        inverse of cdf function - get values whose cumulative density are the given quantiles
        
        args:
            x: tensor of any shape with values in range (0,1)
        return:
            values at which cumulative density are x
        """
        raise NotImplementedError()
    
    def _quantize(self, x, mode):
        """
        quantize representation to discrete value
        
        "noise" mode for training, uniform(-.5,.5) noise is added to x.
        "quantize" mode for evaluating perfomance only, return round(x+0.5).
        "symbol" mode to actually encode given values, return round(x+0.5-mean)
        
        args
            --mode(str): either "noise" of or "quantize" or "symbol"
            --x: tensor of shape #########
        """
        half = self.bin / 2
        if mode == "noise":
            noise = torch.rand_like(x) - half
            noise.detach_()
            return x + noise
        elif mode == "quantize":
            return torch.round(x)
        elif mode == "symbol":
            raise NotImplementedError()
    
    def _prob_mass(self, x):
        """
        take prob mass in the left tail as the distribution is symmetric
        """
        half = self.bin / 2
        
        x = x - self._mean
        x = torch.abs(x)
        
        upper = self._standardized_cumulative((half-x)/self._scale)
        lower = self._standardized_cumulative((-half-x)/self._scale)
        return upper - lower


@ENTROPY_MODEL_REGISTRY.register()
class GaussianConditionalModel(SymmetricConditionalModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gauss = torch.distributions.normal.Normal(0.,1.)
    
    def _standardized_cumulative(self, x):
        return self.gauss.cdf(x)
    
    def _standardized_quantile(self, x):
        return self.gauss.icdf(x)


@ENTROPY_MODEL_REGISTRY.register()
class LaplacianConditionalModel(SymmetricConditionalModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.laplace = torch.distributions.laplace.Laplace(0., 1.)
    
    def _standardized_cumulative(self, x):
        return self.laplace.cdf(x)
    
    def _standardized_quantile(self, x):
        return self.laplace.icdf(x)