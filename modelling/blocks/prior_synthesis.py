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
import torch
import math


class HyperpriorSynthesisTransform(nn.Module):
    """
    adapted from https://github.com/tensorflow/compression/blob/master/models/bmshj2018.py
    """
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
            conv = nn.ConvTranspose2d(
                _in_channels, _out_channels, kernel, stride=stride, 
                padding=kernel//2, bias=True, output_padding=stride-1)
            
            # init weights and bias
            nn.init.xavier_normal_(conv.weight.data, math.sqrt(2))
            nn.init.constant_(conv.bias.data, 0.01)
            
            layers.append(conv)
            if i < len(strides) - 1:
                layers.append(nn.ReLU())        
        
        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        NOTE: pytorch implementation used exp as final activation
        """
        return torch.clamp(self._layers(x).exp(), 1e-10, 1e10)
    
            