import torch.nn as nn


class HyperpriorAnalysisTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        strides = cfg.MODEL.HYPER_PRIOR.STRIDES
        kernels = cfg.MODEL.HYPER_PRIOR.KERNELS
        in_channels = cfg.MODEL.LATENT_CHANNELS
        inter_channels = cfg.MODEL.INTER_CHANNELS
        layers = []
        for i, (stride, kernel) in enumerate(zip(strides, kernels)):
            use_bias = True if i < len(strides) - 1 else False
            _in_channels = in_channels if i == 0 else inter_channels
            layers.append(nn.Conv2d(
                _in_channels, inter_channels, kernel, 
                stride=stride, padding=kernel//2, bias=use_bias))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
        
        # init weights and bias
        
        
        self._layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self._layers(x)
            