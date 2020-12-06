import random

import PIL

import numpy as np
import torch
import torchvision.transforms.functional as F
from .auto_augment import RandAugmentCutout, RandAugment, Cutout

def get_transforms(cfg, aug_name):
    """
    """
    augment_list = _get_augment_list(aug_name, cfg)
    transforms = Compose([
        *augment_list,
        ToTensor(),
        ChannelFirst(),
        Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    ])

    return transforms

def _get_augment_list(name, cfg):
    if name == "val":
        return []
    elif name == "train":
        return [
            RandomCrop(size=cfg.DATA.SIZE)
        ]


class Compose:
    def __init__(self, fn_list):
        self.fns = fn_list
    
    def __call__(self, image):
        """
        image: PIL image
        box: Boxes
        """
        for fn in self.fns:
            image = fn(image)
        return image

    def __repr__(self):
        return "\n".join([str(fn) for fn in self.fns])


class ToTensor:
    # def __init__(self, channel_first=True):
        # self.channel_first = channel_first

    def __call__(self, image):
        """
        convert PIL image or numpy array of shape (3,H,W) to torch tensor of shape (3,H,W) with type of float
        args:
            image: PIL image or numpy array of shape (3,H,W) or (1,H,W)
        """
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        # elif isinstance(image, np.ndarray):
        #     assert image.shape[0] in (1,3), f"invalid channel, got {image.shape[0]}"
        
        return torch.from_numpy(image).float()


class ToPILImage:
    def __call__(self, image):
        assert image.dtype == np.uint8
        return PIL.Image.fromarray(image)


class ChannelFirst:
    def __call__(self, image):
        """
        permute channel first
        image : numpy.array, or torch tensor
        return torch tensor
        """
        assert isinstance(image, (torch.Tensor, np.ndarray))

        if isinstance(image, np.ndarray):
            image = torch.tensor(image)
        shape = image.size()
        if len(shape) == 2:
            return image.unsqueeze(0)
        
        image = image.permute(2,0,1)
        return image
        
class Normalize:
    """
    normalize tensor image of shape (C,H,W)
    divide intensity by 255 then normalize by the given mean and std
    NOTE: mean, std are for intensity in range 0...1
    """
    def __init__(self, mean=0., std=1.,):
        """
        mean: float or tuple(float) for each channel
        std: float or tuple(float) for each channel
        """
        assert max(mean) <= 1. and max(std) <= 1.
        self.mean = mean
        self.std = std

        
    def __call__(self, image):
        """
        normalize tensor image of shape (C,H,W)
        divide intensity by 255 then normalize by the given mean and std
        NOTE: mean, std are for intensity in range 0...1
        """
        assert len(self.mean) == len(image)
        image = F.normalize(image/255., self.mean, self.std)

        return image
    

class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):
        """
        image: PIL image
        """
        if random.random() < self.prob:
            image = image.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
        return image

class RandomCrop:
    """
    not padding for under-sized image
    """
    def __init__(self, size, prob=1.,):
        """
        size: int or (w,h)
        """
        if isinstance(size, int):
            size = (size, size)
        self.size = size
    
    def __call__(self, image):
        """
        currently not support padding
        """
        w,h = image.size
        crop_w, crop_h = self.size
        assert w >= crop_w and h >= crop_h, \
            f"got under-sized image {w}x{h} for crop size {crop_w}x{crop_h}"
        x = np.random.randint(w-crop_w)
        y = np.random.randint(h-crop_h)
        img = img.crop((x, y, x+crop_w, y+ crop_h))
        return img


class Translate:
    def __init__(self, prob=0.5, dis=0.125):
        """
        dis: max tranlate distance
        """
        self.prob = prob
        self.dis = dis
    
    def __call__(self, image):
        if random.random() < self.prob:
            x_offset, y_offset = np.random.uniform(low=-self.dis, high=self.dis, size=2)
            w, h = image.size
            x_offset = int(w*x_offset)
            y_offset = int(h*y_offset)
            image = image.transform(image.size, PIL.Image.AFFINE, (1, 0, x_offset, 0, 1, y_offset))
        return image
