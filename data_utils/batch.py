import torch
import math
import numpy as np
from .utils import mixup

class Batch:
    """
    """
    def __init__(self, data, cfg):
        image_ids, imgs = list(zip(*data))
        self.image_ids = image_ids
        self.imgs = torch.stack(list(imgs))
        # self.labels = torch.stack(list(labels))
        # self.cfg = cfg

    def to(self, device):
        self.imgs = self.imgs.to(device)
        # self.labels = self.labels.to(device)

    # def mixup(self):
    #     if np.random.uniform() < self.cfg.DATA.AUG.MIXUP_PROB:
    #         self.imgs, self.labels = mixup(
    #             self.imgs, 
    #             self.labels, 
    #             self.cfg.DATA.AUG.MIXUP_ALPHA, 
    #             self.cfg.DATA.AUG.MIXUP_DIM
    #         )
    
    def pin_memmory(self):
        self.imgs = self.imgs.pin_memory()
        # self.labels = self.labels.pin_memory()

    # def __getitem__(self, idx):
    #     """
    #     return (study_id, img)
    #     """
    #     return (
    #         self.study_ids[idx],
    #         self.images[idx],
    #     )

    # def __len__(self):
    #     return len(self.images)

    # def _padding(self, imgs):
    #     """
    #     padding imgs to get list of images of th same shape
    #     NOTE: padding to the right and bottom of the image

    #     """
    #     max_h, max_w = np.max([img.size()[-2:] for img in imgs], axis=0)
    #     max_h = self.division * math.ceil(max_h/self.division)
    #     max_w = self.division * math.ceil(max_w/self.division)

    #     N = len(imgs)
    #     C = imgs[0].size(0)
    #     batch_imgs = torch.zeros((N, C, max_h, max_w), dtype=torch.float32)

    #     for i, img in enumerate(imgs):
    #         h, w = img.size()[-2:]
    #         batch_imgs[i, :, :h, :w] = img

    #     return batch_imgs

class UnlabelBatch:
    def __init__(self, data):
        image_ids, w_imgs, str_imgs = list(zip(*data))
        self.image_ids = image_ids
        self.w_imgs = torch.stack(list(w_imgs))
        self.str_imgs = torch.stack(list(str_imgs))
        # print("unlabel_batch")
    def to(self, device):
        self.w_imgs = self.w_imgs.to(device)
        self.str_imgs = self.str_imgs.to(device)

    def pin_memmory(self):
        self.w_imgs = self.w_imgs.pin_memory()
        self.str_imgs = self.str_imgs.pin_memory()
