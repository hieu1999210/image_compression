import torch.nn.funcional as F
import math

LOG10 = math.log(10)


def ms_ssmi(img1, img2):
    pass

def psnr(img1, img2, max_val=255.):
    """
    computer by batch
    
    image: shape (N,H,W,C), or (N,C,H,W)
    compute in dB psnr = 10log
    """
    
    mse = F.mse_loss(img1, img2, reduction="none").mean((1,2,3))
    
    return 10. * (2*math.log10(max_val) - mse.log()/LOG10)
    
    


