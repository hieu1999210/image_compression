"""
from detectron2
"""
from torch.utils.data.sampler import Sampler
import torch


class TrainingSampler(Sampler):
    def __init__(self, size, shuffle, seed=None):
        self._size = size
        self._shuffle = shuffle
        self._seed = int(seed)
    
    def __iter__(self):
        yield from self._infinite_indices()
    
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

if __name__ == "__main__":
    from torch.utils.data import BatchSampler
    a = TrainingSampler(10, True, 0)
    batch_sampler = BatchSampler(a, 3, drop_last=True)
    x =iter(batch_sampler)
    print(next(x))
    print(next(x))