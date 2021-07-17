# Copyright 2020 Hieu Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from glob import glob 

from PIL import Image
from torch.utils.data import Dataset

from ..transforms import get_transforms
from .build import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class KodakDataset(Dataset):

    def __init__(self, data_folder, mode, cfg, **kwargs):
        """
        """
        super().__init__()

        self.cfg = cfg
        self.paths = sorted(glob(f"{data_folder}/*"))
        print(f"There are {len(self)} image in {mode} dataset")

        self.transforms = get_transforms(cfg, mode)

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        """
        """
        path = self.paths[idx]
        image_id = os.path.split(path)[-1].replace(".png", "")
        img = self._load_img(idx)
        img = self.transforms(img)

        return image_id, img

    def _load_img(self, idx):
        """
        args: image path
        return: pillow image
        """
        image = Image.open(self.paths[idx]).convert('RGB')

        return image
    

