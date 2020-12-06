from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
import matplotlib.pyplot as plt

from ..transforms import get_transforms
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetDataset(Dataset):

    def __init__(self, data_folder, mode, cfg, **kwargs):
        """
        """
        super().__init__()

        self.cfg = cfg
        self.data_folder = data_folder
        self.paths = sorted(glob(f"{data_folder}/*/*"))
        print(f"There are {len(self)} image in {mode} dataset")

        self.transforms = get_transforms(cfg, mode)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        """
        """
        path = self.paths[idx]
        image_id = os.path.split(path)[-1].replace(".jpg", "")
        img = self._load_img(path)
        
        img = self.transforms(img)
            return image_id, img

    def _load_img(self, path):
        """
        args: image path
        return: pillow image
        """
        image = Image.open(img_path).convert('RGB')

        return image