"""
from https://github.com/facebookresearch/detectron2
"""

# Copyright (c) Facebook, Inc. and its affiliates.
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

import torch
from torch.utils.data.sampler import Sampler


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
    x = iter(batch_sampler)
    print(next(x))
    print(next(x))