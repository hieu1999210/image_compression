import torch
import math
import numpy as np


class Batch:
    """
    """
    def __init__(self, data, cfg):
        image_ids, imgs = list(zip(*data))
        self.image_ids = image_ids
        self.imgs = torch.stack(list(imgs))

    def to(self, device):
        self.imgs = self.imgs.to(device)

    def pin_memmory(self):
        self.imgs = self.imgs.pin_memory()
