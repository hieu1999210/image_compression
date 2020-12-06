from torch.utils.data import DataLoader

from utils import Registry
DATALOADER_REGISTRY = Registry("DATALOADER")


def build_dataloader(cfg, **kwargs):
    dataloader_name = cfg.DATA.DATALOADER_NAME
    dataloader = DATALOADER_REGISTRY.get(dataloader_name)(cfg=cfg, **kwargs)
    assert isinstance(dataloader, DataLoader)

    return dataloader