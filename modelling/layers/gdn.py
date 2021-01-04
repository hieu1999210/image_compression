import torch.nn.functional as F
import torch.nn as nn
import torch
from .bound import LowerBound


class NonNegativeParam(nn.Module):
    """
    adapted from https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/parameterizers.py
    
    non negative parameter wraper for GDN
    """
    def __init__(self, init_val, minimum=0, offset=2**(-18)):
        super().__init__()
        offset = float(offset)
        minimum = float(minimum)
        self.pedestal = torch.tensor(offset**2)
        self.bound = (minimum + self.pedestal.clone().detach()**2)**0.5
        
        self.param = nn.Parameter(torch.sqrt(torch.max(
            init_val+self.pedestal, self.pedestal)))
        self.register_parameter("param", self.param)
        
    def forward(self):
        val = LowerBound.apply(self.param, self.bound)
        val = val*val - self.pedestal
        return val


class GDN(nn.Module):
    def __init__(self, in_channels, inverse=False, relu=False, 
        gamma_init=0.1, beta_min=1e-6, offset=2**-18,
    ):
        super().__init__()
        self.inverse = inverse
        self.relu = relu
        self.gamma = NonNegativeParam(
            torch.eye(in_channels).view(in_channels, in_channels, 1, 1)*gamma_init)
        self.beta = NonNegativeParam(torch.ones(((in_channels,))), minimum=beta_min)

    def forward(self, x):
        if self.relu:
            x = F.relu(x)
        gamma = self.gamma()
        beta = self.beta()
        norm = torch.sqrt(F.conv2d(x*x, gamma , beta))
        
        x = x * norm if self.inverse else x / norm
        
        return x