from torch.utils.data import Dataset

from utils import Registry
DATASET_REGISTRY = Registry("DATASET")

def build_dataset(dataset_name, cfg, **kwargs):
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg=cfg, **kwargs)
    assert isinstance(dataset, Dataset)

    return dataset