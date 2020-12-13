import torch.nn.functional as F
import torch
import torch.nn as nn
from .layers import LowerBound


def get_loss_dict(cfg, names):
    lcfg = cfg.MODEL.LOSS
    loss_fn = {
        "MSE": nn.MSELoss(reduction=cfg.MODEL.LOSS.REDUCTION),
        "SSIMLoss": SSIMLoss(
            max_val=lcfg.SSIM.MAX_VAL,
            filter_size=lcfg.SSIM.FILTER_SIZE,
            filter_sigma=lcfg.SSIM.FILTER_SIGMA,
            k1=lcfg.SSIM.K1,
            k2=lcfg.SSIM.K2,
            log_scale=lcfg.SSIM.LOG_SCALE,
            eps=lcfg.SSIM.EPS,
        ),
        "MS_SSIMLoss": MS_SSIMLoss(
            max_val=lcfg.SSIM.MAX_VAL,
            filter_size=lcfg.SSIM.FILTER_SIZE,
            filter_sigma=lcfg.SSIM.FILTER_SIGMA,
            k1=lcfg.SSIM.K1,
            k2=lcfg.SSIM.K2,
            log_scale=lcfg.SSIM.LOG_SCALE,
            eps=lcfg.SSIM.EPS,
        ),
    }
    return {name: loss_fn[name] for name in names}


class SSIMLoss(nn.Module):
    """
    assume input images have intensity in range [0,1]
    and lower bound 0 is set for ssim
    ssim loss is calulated as 1 - ssim
    """
    def __init__(self, 
        max_val=255., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, log_scale=False, eps=1e-5
    ):
        super().__init__()
        self.max_val = max_val
        self.filter = self._make_filter(filter_size, filter_sigma)
        self.c1 = (k1*max_val)**2
        self.c2 = (k2*max_val)**2
        self.log_scale = log_scale
        self.eps = eps
    def forward(self, img1, img2):
        # rescale imgs to the given range
        img1 = img1 * self.max_val
        img2 = img2 * self.max_val
        
        ssim, _ = self._ssim(img1, img2)
        if self.log_scale:
            return -ssim
        return 1. - ssim.mean()
    
    def _ssim(self, img1, img2):
        """
        image of shape N,C,H,W
        return ssim and cs of shape N,C,H',W'
        """
        
        mu1, mu2 = self._gauss_filter(img1), self._gauss_filter(img2)
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self._gauss_filter(img1*img1) - mu1_sq
        sigma2_sq = self._gauss_filter(img2*img2) - mu2_sq
        sigma12 = self._gauss_filter(img1*img2) - mu1_mu2

        cs = (2. * sigma12 + self.c2) / (sigma1_sq + sigma2_sq + self.c2)
        ssim = cs * (2. * mu1_mu2 + self.c1) / (mu1_sq + mu2_sq + self.c1)
        
        ssim, cs = ssim.mean((1,2,3)), cs.mean((1,2,3))
        
        # set lowwer bound
        eps = self.eps if self.log_scale else 0.
        ssim = LowerBound.apply(ssim, eps)
        cs = LowerBound.apply(cs, eps)
        if self.log_scale:
            return ssim.log(), cs.log()
        return ssim, cs
    
    def _gauss_filter(self, img):
        """
        img shape N,C,H,W
        """
        N,C,H,W = img.size()
        img = img.view(-1,1,H,W)
        img = F.conv2d(img, self.filter)
        _, _, H, W = img.size()
        return img.view(N,C,H,W)
    
    def _make_filter(self, filter_size, filter_sigma):
        """
        make 2d gaussian kernel with given size and sigma
        """
        value_range = torch.arange(filter_size) +.5 - filter_size/2
        x, y = torch.meshgrid(value_range, value_range)
        g = - (x*x+y*y) / (2. * filter_sigma * filter_sigma)
        g = F.softmax(g.view(-1), dim=0).view(1,1,filter_size,filter_size)
        return g


class MS_SSIMLoss(SSIMLoss):
    def __init__(self, 
        max_val=255., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, 
        log_scale=False, eps=1e-5, weights=[0.0448,0.2856,0.3001,0.2363,0.1333],
    ):
        super().__init__(max_val, filter_size, filter_sigma, k1, k2, log_scale, eps)
        self.weights = weights
    
    def forward(self, img1, img2):
        # rescale imgs to the given range
        img1 = img1 * self.max_val
        img2 = img2 * self.max_val
        
        if self.log_scale:
            return self._log_forward(img1, img2)
        return self._forward(img1, img2)
    
    def _log_forward(self, img1, img2):
        """
        forward for log scale
        assume imgs are rescaled to the correct intensity range
        """
        results = 0.
        n_levels = len(self.weights)
        for i, weight in enumerate(self.weights, 1):
            ssim, cs = self._ssim(img1, img2)
            
            if i < n_levels:
                results = results + cs*weight
                img1, img2 = self.downsample(img1), self.downsample(img2)
            else:
                results = results + ssim*weight
        return - results.mean()
    
    def _forward(self, img1, img2):
        """
        forward for orignal version of ms ssim
        assume imgs are rescaled to the correct intensity range
        """
        N = img1.size(0)
        results = torch.ones((N,), dtype=img1.dtype).to(img1.device)
        n_levels = len(self.weights)
        for i, weight in enumerate(self.weights, 1):
            ssim, cs = self._ssim(img1, img2)
            
            if i < n_levels:
                results = results * cs**weight
                img1, img2 = self.downsample(img1), self.downsample(img2)
            else:
                results = results * ssim**weight
        return 1. - results.mean()
    
    def downsample(self, img):
        """
        downsample img by 2
        """
        # reflection pad to img so that img shape is divisible by 2
        _, _, h, w = img.size()
        pad_x = w%2
        pad_y = h%2
        if pad_x + pad_y:
            pad_fn = nn.ReflectionPad2d((0,pad_x,0,pad_y))
            img = pad_fn(img)
        
        return F.avg_pool2d(img, kernel_size=2)
