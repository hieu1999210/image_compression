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

import torch
import torch.nn as nn
from ..blocks import (
    AnalysisTransform,
    HyperpriorAnalysisTransform,
    HyperpriorSynthesisTransform,
    SynthesisTransform,
    ENTROPY_MODEL_REGISTRY,
)
from ..layers import LowerBound, UpperBound
from .build import META_ARCH_REGISTRY
from ..loss import get_loss_dict


@META_ARCH_REGISTRY.register()
class Compressor2018(nn.Module):
    """
    adapted from https://github.com/tensorflow/compression/blob/master/models/bmshj2018.py
    """
    def __init__(self, cfg):
        super().__init__()
        self.analysis_transform     = AnalysisTransform(cfg)
        self.prior_analysis         = HyperpriorAnalysisTransform(cfg)
        self.prior_synthesis        = HyperpriorSynthesisTransform(cfg)
        self.synthesis_transform    = SynthesisTransform(cfg)
        self.entropy_model          = ENTROPY_MODEL_REGISTRY.get("EntropyModel")(
            cfg.MODEL.LATENT_CHANNELS, cfg)
        self.conditional_model      = ENTROPY_MODEL_REGISTRY.get(
            cfg.MODEL.ENTROPY_MODEL.CONDITIONAL_MODEL)(cfg)
        self.distortion_loss_fns    = get_loss_dict(cfg, cfg.MODEL.LOSS.DISTORTION_LOSS_NAMES)
        self.distortion_loss_weight = cfg.MODEL.LOSS.DISTORTION_LOSS_WEIGHT
        self.loss_names = ["y_entropy", "z_entropy", "bpp"] + list(self.distortion_loss_fns.keys())
        
    def forward(self, x):
        N,C,H,W = x.size()
        num_pixels = N*H*W
        y = self.analysis_transform(x)
        z = self.prior_analysis(torch.abs(y))
        z_tilde, z_probs, z_ce_loss= self.entropy_model(z)
        
        sigma = self.prior_synthesis(z_tilde)
        y_tilde, y_probs = self.conditional_model(y, sigma)
        y_ce_loss = self.conditional_model._ce_loss(y_probs)
        
        x_tilde = self.synthesis_transform(y_tilde)
        x_tilde = UpperBound.apply(x_tilde, 1.)
        x_tilde = LowerBound.apply(x_tilde, 0.)
        
        # # distortion loss
        distortion_losses = self.distortion_loss(x, x_tilde)
        total_distortion_loss = sum(distortion_losses.values())
        distortion_losses = {k:v.detach() for k,v in distortion_losses.items()}
        # entropy loss
        entropy_loss = (z_ce_loss + y_ce_loss) / num_pixels
        
        total_loss = self.distortion_loss_weight*total_distortion_loss + entropy_loss
        losses = {
            "z_entropy": z_ce_loss.detach() / num_pixels,
            "y_entropy": y_ce_loss.detach() / num_pixels,
            "bpp": entropy_loss.detach(),
            "total_loss": total_loss,
        }
        losses.update(distortion_losses)
        return x_tilde.detach(), losses

    def distortion_loss(self, img1, img2):
        losses = {}
        for name, loss_fn in self.distortion_loss_fns.items():
            losses[name] = loss_fn(img1, img2)
        return losses
        
    def compress(self, x):
        pass
    
    def decompress(self, x):
        pass
