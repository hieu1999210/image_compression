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
    """
    adapted from https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py
    """
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