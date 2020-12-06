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
    # not for cifar dataset
    data_folder = cfg.DIRS.TRAIN_DATA if mode == "train" else cfg.DIRS.VAL_DATA

    dataset = build_dataset(
        cfg=cfg,
        data_folder=data_folder,
        mode=mode,
    )

    batch_size = {
        "val": cfg.VAL.BATCH_SIZE,
        "train": cfg.SOLVER.IMS_PER_BATCH,
    }[mode]
    
    collate_fn = lambda x: Batch(x, cfg)

    if mode = "train":
        sampler = TrainingSampler(len(dataset), True, cfg.SEED)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
        return DataLoader(
            dataset, batch_sampler=batch_sampler, 
            num_workers=cfg.DATA.NUM_WORKERS, collate_fn=collate_fn)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, 
        num_workers=cfg.DATA.NUM_WORKERS, collate_fn=collate_fn)