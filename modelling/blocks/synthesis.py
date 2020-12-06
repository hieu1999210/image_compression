import torch.nn as nn
from ..layers import GDN
import math

class SynthesisTransform(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        strides = cfg.MODEL.STRIDES
        kernel = cfg.MODEL.CONV_KERNEL
        out_channels = cfg.DATA.IN_CHANNELS
        inter_channels = cfg.MODEL.INTER_CHANNELS
        latent_channels = cfg.MODEL.LATENT_CHANNELS
        layers = []
        for i, stride in enumerate(strides):
            _in_channels = latent_channels if i == 0 else inter_channels
            _out_channels = out_channels if i == len(strides) - 1 else inter_channels
            conv = nn.ConvTranspose2d(
                _in_channels, _out_channels, kernel, 
                stride=stride, padding=kernel//2, output_padding=stride-1)
            
            # init weights and bias
            nn.init.xavier_normal_(conv.weight.data, math.sqrt(2))
            nn.init.constant_(conv.bias.data, 0.01)
            
            layers.append(conv)
            if i < len(strides) -1:
                gdn = GDN(_out_channels)
                layers.append(gdn)
            
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)