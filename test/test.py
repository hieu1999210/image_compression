import cv2
import os
from glob import glob
from utils import psnr, SSIM, MS_SSIM
import torch
from modelling import MS_SSIMLoss, SSIMLoss
folder = "/home/hieu/pred"
dest_folder = "/home/hieu/kodak"
dest2_folder = "/home/hieu/kodak_pred"
# for img_path in glob(f"{folder}/*"):
#     img = cv2.imread(img_path)
#     H,W,C = img.shape
#     W = int((W-10) / 2)
#     assert W in (512,768)
#     gt = img[:,:W,:]
#     dest_path = img_path.replace(folder, dest_folder)
#     cv2.imwrite(dest_path, gt)
    
#     pred = img[:,W+10:,:]
#     dest_path = dest_path.replace(dest_folder, dest2_folder)
#     cv2.imwrite(dest_path, pred)
ssim_fn = SSIM(device="cpu", in_dB=True)
ms_ssim_fn = MS_SSIM(device="cpu", in_dB=True)
ms_ssim_loss = MS_SSIMLoss(log_scale=True)
results = 0.0
results2 = 0.0
loss = 0.
results3 = 0.
for i in range(1,25):
    gt = cv2.imread(os.path.join(dest_folder, f"{i}.png"))
    pred = cv2.imread(os.path.join(dest2_folder, f"{i}.png"))
    gt = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float()
    pred = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).float()
    # results = results + psnr(pred, gt).mean()
    ssim = ssim_fn(pred, gt)
    ms_ssim = ms_ssim_fn(pred, gt)
    l = ms_ssim_loss(pred/255., gt/255.)
    results2 += ssim
    results3 += ms_ssim
    loss += l
    print(i, ssim, ms_ssim, l)


# print(results/24)
print(results2/24)
print(results3/24)
print(loss/24)