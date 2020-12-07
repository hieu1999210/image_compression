from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from ..transforms import get_transforms
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetDataset(Dataset):

    def __init__(self, data_folder, metadata, mode, cfg, **kwargs):
        """
        """
        super().__init__()

        self.cfg = cfg
        self.data_folder = data_folder
        self.paths = pd.read_csv(metadata)["path"].tolist()
        print(f"There are {len(self)} image in {mode} dataset")

        self.transforms = get_transforms(cfg, mode)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        """
        """
        path = self.paths[idx]
        image_id = os.path.split(path)[-1].replace(".jpg", "")
        img = self._load_img(idx)
        img = self.transforms(img)

        return image_id, img

    def _load_img(self, idx):
        """
        args: image path
        return: pillow image
        """
        path = os.path.join(self.data_folder, self.paths[idx])
        image = Image.open(path).convert('RGB')

        return image
    
