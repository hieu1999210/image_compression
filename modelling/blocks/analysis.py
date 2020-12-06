import torch.nn as nn
from ..layers import GDN

class AnalysisTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        strides = cfg.MODEL.STRIDES
        kernel = cfg.MODEL.CONV_KERNEL
        in_channels = cfg.DATA.IN_CHANNELS
        inter_channels = cfg.MODEL.INTER_CHANNELS
        latent_channels = cfg.MODEL.LATENT_CHANNELS
        layers = []
        for i, stride in enumerate(strides):
            _in_channels = in_channels if i == 0 else inter_channels
            _out_channels = latent_channels if i == len(strides) - 1 else inter_channels
            conv = nn.Conv2d(_in_channels, _out_channels, kernel, stride=stride, padding=kernel//2)
            layers.append(conv)
            if i < len(strides) -1:
                gdn = GDN(_out_channels)
                layers.append(gdn)
            ### missing init weights
            
            
            
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)