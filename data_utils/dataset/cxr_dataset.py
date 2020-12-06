from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from ..transforms import get_transforms
from .build import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class CXR(Dataset):

    def __init__(self, data_folder, mode, cfg, metadata, **kwargs):
        """
        mode: unlabeled - unlabeled training set
              labeled - labeled training set
              val - validation set
        """
        super().__init__()
        assert mode in ["train", "val"]

        self.cfg = cfg
        self.data_folder = data_folder
        self.mode = mode
        metadata = pd.read_csv(metadata)
        self.labels = metadata[cfg.DATA.CXR_DISEASES].values
        self.ids = metadata["imageUid"].values#[:256]


        if mode == "val":
            self.transforms = get_transforms(cfg, "val")
        elif mode == "train":
            self.transforms = get_transforms(cfg, cfg.DATA.AUG.SUPERVISED)
        
        print(f"There are {len(self)} image in {mode} dataset")

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        """
        image_id = self.ids[idx]

        img = self._load_img(image_id)
        img = self.transforms(img)
        label = torch.tensor(self.labels[idx]).float()
        return image_id, img, label

    def _load_img(self, idx):
        """
        args: image path
        return: pillow image
        """

        img_path = os.path.join(self.data_folder, f"{idx}.png")
        # img_path = os.path.join(self.data_folder, f"{idx}")
        image = Image.open(img_path)
        # Convert 1-channel to 3-channels 
        if self.cfg.DATA.IN_CHANNELS == 3:
            image = image.convert('RGB')

        return image
