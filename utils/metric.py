import math

import torch.nn.functional as F
import torch
import torch.nn as nn

LOG10 = math.log(10)


def psnr(img1, img2, max_val=255.):
    """
    computer by batch
    
    image: shape (N,H,W,C), or (N,C,H,W)
    compute in dB psnr = 10log
    """
    
    mse = F.mse_loss(img1, img2, reduction="none").mean((1,2,3))
    
    return 10. * (2*math.log10(max_val) - mse.log()/LOG10)


class SSIM:
    def __init__(self, 
        max_val=255., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, device="cuda",
        in_dB=False, non_negative=False,
    ):
        self.filter = self._make_filter(filter_size, filter_sigma).to(device)
        self.c1 = (k1*max_val)**2
        self.c2 = (k2*max_val)**2
        self.non_negative = non_negative
        self.in_dB = in_dB
        
    def __call__(self, img1, img2):
        ssim, _ = self._ssim(img1, img2)
        if self.in_dB:
            ssim = - 10. * (1.-ssim).log() / LOG10
        return ssim
    
    def _ssim(self, img1, img2):
        """
        image of shape N,C,H,W
        return ssim and cs of shape N,C,H',W'
        """
        mu1, mu2 = self._gauss_filter(img1), self._gauss_filter(img2)
        # print(mu1.min(), mu2.min())
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self._gauss_filter(img1*img1) - mu1_sq
        sigma2_sq = self._gauss_filter(img2*img2) - mu2_sq
        sigma12 = self._gauss_filter(img1*img2) - mu1_mu2
        # print(sigma1_sq.min(), sigma2_sq.min())
        # print(sigma12.min())
        # contrast and scale similarity
        cs = (2. * sigma12 + self.c2) / (sigma1_sq + sigma2_sq + self.c2)
        
        ssim = cs * (2. * mu1_mu2 + self.c1) / (mu1_sq + mu2_sq + self.c1)
        
        ssim, cs = ssim.mean((1,2,3)), cs.mean((1,2,3))
        
        if self.non_negative:
            torch.clamp_min_(ssim, 0.)
            torch.clamp_min_(cs, 0.)
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


class MS_SSIM(SSIM):
    def __init__(self, 
        max_val=255., filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, 
        device="cuda", in_dB=True, weights=[0.0448,0.2856,0.3001,0.2363,0.1333],
    ):
        super().__init__(
            max_val, filter_size, filter_sigma, k1, k2, device, non_negative=True, in_dB=False)
        self.in_dB = in_dB
        self.weights = weights
    
    def __call__(self, img1, img2):
        N = img1.size(0)
        results = torch.ones((N,), dtype=img1.dtype).to(img1.device)
        n_levels = len(self.weights)
        for i, weight in enumerate(self.weights, 1):
            ssim, cs = self._ssim(img1, img2)
            
            if i < n_levels:
                # print(results.device)
                # print(ssim.device)
                # print(weight.device)
                results = results * cs**weight
                img1, img2 = self.downsample(img1), self.downsample(img2)
            else:
                # print(results.device)
                # print(ssim.device)
                # print(weight.device)
                results = results * ssim**weight
        if self.in_dB:
            results = - 10. * (1.-results).log() / LOG10
        return results
    
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

        
if __name__ == "__main__":
    obj = SSIM(device="cpu")
    obj2 = MS_SSIM(device="cpu", in_dB=False)
    img1 = torch.rand(2,3,256,256)*255
    img2 = torch.rand(2,3,256,256)*255
    print(obj(img1,img2))
    print(obj2(img1,img2))