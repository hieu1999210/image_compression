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

from torch.utils.data import DataLoader, BatchSampler

from .sampler import TrainingSampler
from .batch import Batch
from .build_dataloader import DATALOADER_REGISTRY
from .dataset import build_dataset


__all__ = [
    "infinite_dataloader",
]


@DATALOADER_REGISTRY.register()
def infinite_dataloader(mode, cfg):
    """
    get dataloader for iteration-based training with infinite sampler
    mode: "train", "val"
    """
    assert mode in ["val", "train"]
    data_folder = cfg.DIRS.TRAIN_DATA if mode == "train" else cfg.DIRS.VAL_DATA
    metadata = cfg.DIRS.TRAIN_METADATA if mode == "train" else None
    dataset_name = cfg.DATA.TRAIN_DATASET_NAME if mode == "train" else cfg.DATA.VAL_DATASET_NAME
    dataset = build_dataset(
        dataset_name=dataset_name,
        cfg=cfg,
        data_folder=data_folder,
        mode=mode,
        metadata=metadata,
    )

    batch_size = {
        "val": cfg.VAL.BATCH_SIZE,
        "train": cfg.SOLVER.IMS_PER_BATCH,
    }[mode]
    
    collate_fn = lambda x: Batch(x, cfg)

    if mode == "train":
        sampler = TrainingSampler(len(dataset), True, cfg.SEED)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        return DataLoader(
            dataset, batch_sampler=batch_sampler, 
            num_workers=cfg.DATA.NUM_WORKERS, collate_fn=collate_fn)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, 
        num_workers=cfg.DATA.NUM_WORKERS, collate_fn=collate_fn)
