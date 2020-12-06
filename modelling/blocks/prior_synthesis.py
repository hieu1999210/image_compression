import torch.nn as nn
import torch


class HyperpriorSynthesisTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        strides = cfg.MODEL.HYPER_PRIOR.STRIDES
        kernels = cfg.MODEL.HYPER_PRIOR.KERNELS
        latent_channels = cfg.MODEL.LATENT_CHANNELS
        inter_channels = cfg.MODEL.INTER_CHANNELS
        layers = []
        for i, (stride, kernel) in enumerate(zip(reversed(strides), reversed(kernels))):
            _in_channels = inter_channels
            _out_channels = inter_channels if i < len(strides) - 1 else latent_channels
            layers.append(nn.ConvTranspose2d(
                _in_channels, _out_channels, kernel, stride=stride, 
                padding=kernel//2, bias=True, output_padding=stride-1))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
        
        # init weights and bias
        
        
        self._layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        NOTE: pytorch implementation used exp as final activation
        """
        return torch.clamp(self._layers(x).exp(), 1e-10, 1e10)
    
            