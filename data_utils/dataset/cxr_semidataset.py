from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from bisect import bisect_right

from ..transforms import get_transforms
from .build import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SemiCXR(Dataset):

    def __init__(self, data_folder, datasets, mode, cfg, **kwargs):
        """
        mode: unlabeled - unlabeled training set
              labeled - labeled training set
              val - validation set
        """
        super().__init__()
        assert mode in ["unlabeled", "labeled", "val"]
        for x in datasets:
            assert x in ["chexpert", "mimic", "vinbdi"]
        self.cfg = cfg
        self.data_folder = data_folder
        self.mode = mode
        self.datasets = datasets

        sizes, ids, labels = [], [], []
        for dataset in datasets:
            _ids, metadata = self._load_metadata(dataset)
            sizes.append(len(_ids))
            ids.append(_ids)
            if mode != "unlabeled":
                labels.append(metadata[cfg.DATA.CXR_DISEASES].values)
        
        ids = np.concatenate(ids)
        self._separate_id = [sum(sizes[:i+1]) for i in range(len(sizes))]
        if mode != "unlabeled":
            self.labels = np.concatenate(labels)
        self.ids = ids

        if mode == "val":
            self.transforms = get_transforms(cfg, "val")
        elif mode == "labeled":
            self.transforms = get_transforms(cfg, cfg.DATA.AUG.SUPERVISED)
        elif mode == "unlabeled":
            self.transforms = {
                "weak": get_transforms(cfg, cfg.DATA.AUG.WEAK), 
                "strong": get_transforms(cfg, cfg.DATA.AUG.STRONG),
            }
        
        print(f"There are {len(self)} image in {mode} dataset")

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        """
        dataset = self._get_dataset_from_idx(idx)
        image_id = self.ids[idx]

        img = self._load_img(image_id, dataset)
        if self.mode == "unlabeled":
            w_img = self.transforms["weak"](img)
            str_img = self.transforms["strong"](img)
            return image_id, w_img, str_img

        img = self.transforms(img)
        label = torch.tensor(self.labels[idx]).float()
        return image_id, img, label

    def _get_dataset_from_idx(self, idx):
        return self.datasets[bisect_right(self._separate_id, idx)]

    def _load_img(self, idx, dataset):
        """
        args: image path
        return: pillow image
        """
        if dataset == "vinbdi":
            img_path = os.path.join(self.data_folder, "vinbdi", "img", f"{idx}.png")
        elif dataset == "chexpert":
            img_path = os.path.join(self.data_folder, "chexpert", "img", f"{idx}")
        elif dataset == "mimic":
            img_path = os.path.join(self.data_folder, "mimic", "img", f"{idx}")
        image = Image.open(img_path)

        # Convert 1-channel to 3-channels 
        if self.cfg.DATA.IN_CHANNELS == 3:
            image = image.convert('RGB')

        return image
    
    def _load_metadata(self, dataset):
        """
        return ids, original_data only used for train
        """
        mode = "val" if self.mode == "val" else "train"
        data = pd.read_csv(os.path.join(self.data_folder, dataset, f"{mode}.csv"))

        ids = data["imageUid"].values if dataset == "vinbdi" else data["Path"].values 

        return ids, data


