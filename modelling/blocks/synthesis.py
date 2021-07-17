"""
modified by Hieu Nguyen
adapted from Tensorflow to Pytorch implementation 
"""
# Copyright 2020 Hieu Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch.nn as nn
from ..layers import GDN
import math


class SynthesisTransform(nn.Module):
    """
    adapted from https://github.com/tensorflow/compression/blob/master/models/bmshj2018.py
    """
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