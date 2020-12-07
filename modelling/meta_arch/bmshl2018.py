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


@META_ARCH_REGISTRY.register()
class Compressor2018(nn.Module):
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
        self.mse_loss               = nn.MSELoss()
        self.distortion_loss_weight = cfg.MODEL.LOSS.DISTORTION_LOSS_WEIGHT
        self.loss_names             = ["y_entropy", "z_entropy", "distortion", "bpp"]
        
    def forward(self, x):
        N,C,H,W = x.size()
        num_pixels = N*H*W
        y = self.analysis_transform(x)
        # print("y", y.shape)
        z = self.prior_analysis(torch.abs(y))
        # print("z", z.shape)
        z_tilde, z_probs, z_ce_loss= self.entropy_model(z)
        
        sigma = self.prior_synthesis(z_tilde)
        y_tilde, y_probs = self.conditional_model(y, sigma)
        y_ce_loss = self.conditional_model._ce_loss(y_probs)
        
        x_tilde = self.synthesis_transform(y_tilde)
        x_tilde = UpperBound.apply(x_tilde, 1.)
        x_tilde = LowerBound.apply(x_tilde, 0.)
        # print("z_tilde", z_tilde.shape)
        # print("y_tilde", y_tilde.shape)
        # print("x_tilde", x_tilde.shape)
        # # distortion loss
        mse_loss = self.mse_loss(x, x_tilde)
        
        # entropy loss
        entropy_loss = (z_ce_loss + y_ce_loss) / num_pixels
        
        total_loss = self.distortion_loss_weight*mse_loss + entropy_loss
        return x_tilde.detach(), {
            "z_entropy": z_ce_loss.detach() / num_pixels,
            "y_entropy": y_ce_loss.detach() / num_pixels,
            "bpp": entropy_loss.detach(),
            "distortion": mse_loss.detach(),
            "total_loss": total_loss,
        }

    def compress(self, x):
        pass
    
    def decompress(self, x):
        pass
