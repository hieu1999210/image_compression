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

import logging
import os
import sys

import numpy as np
import torch


def get_log(name, folder, rank=0, file_name='logs.log', console=True):
    
    assert os.path.isdir(folder), f'log dir \'{folder}\' does not exist.'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if rank > 0:
        return logger
    
    log_format = logging.Formatter(
        '{asctime}:{name}:  {message}',
        style='{'
    )
    if folder:
        fh = logging.FileHandler(os.path.join(folder, file_name))
        fh.setFormatter(log_format)
        logger.addHandler(fh)
    
    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(log_format)
        logger.addHandler(ch)
        
    return logger


def write_stats(csv_writer, mode='train', **stats):
    """
    write stats to csv file
    """
    for key, value in stats.items():
        name = mode + '_' + key
        if isinstance(value, list):
            csv_writer.writerow([name]+value)
        else:
            csv_writer.writerow([name, value])


def write_log(_print, mode='Train', **info):
    """
    write_log using logging
    args:
        --_print: usualy is logging.info
    """
    message = "\n"
    for key, value in info.items():
        message += '{:6s} {:25s}: {:.4f}\n'.format(mode, key, value)
    _print(message)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, cache=True, dtype=float):
        self.cache = cache
        self.reset()
        self.dtype = dtype
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.array = [] if self.cache else None

    def update(self, val, n=1):
        if not isinstance(val, self.dtype):
            val = self.dtype(val)
            
        self.val = val
        if self.cache: self.array.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.val:09.6f}({self.avg:09.6f})"


class LogitAverageMeter:
    """ensemble for both seg and cls"""

    def __init__(self, folder, study_ids, num_checkpoints=10, dataset="Validation"):
        self.count = {x:0 for x in study_ids}
        self.num_checkpoints = num_checkpoints
        self.folder = folder
        self.dataset = dataset

    def update(self, data, study_id):
        n = self.num_checkpoints
        assert self.count[study_id] < n

        path = self._get_path(study_id)
        if self.count[study_id] == 0:
            current_data = {k: 0.0 for k in data.keys()}
        else:
            current_data = np.load(path, allow_pickle=True).item()
        
        # update
        current_data = {k: logit+data[k] for k, logit in current_data.items()}
        self.count[study_id] += 1

        if self.count[study_id] == n:
            current_data = {k: v/float(n) for k,v in current_data.items()}
        
        np.save(path, current_data)
    
    def _get_path(self, study_id):
        return os.path.join(
            self.folder,
            f"BraTS20_{self.dataset}_{study_id:0>3}_seg.npy"
        )


class BatchAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, cache=True):
        self.cache = cache
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.array = None

    def update(self, values):
        
        assert isinstance(values, (torch.Tensor, np.ndarray)), \
            f"expect value of type torch.Tensor, np.ndarray but got {type(values)}"
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        assert len(values.shape) == 1
        
        self.val = values.mean()
        
        if self.cache: 
            if self.array is None:
                self.array  = values
            else:
                self.array = np.concatenate([self.array, values], axis=0)
        self.sum += values.sum()
        self.count += len(values)
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.val:.6f}({self.avg:.6f})"


class Storer(object):
    """item for each iter"""

    def __init__(self, dtype="array"):
        """
        expect each item is either np.ndarray or torch.tensor
        args:
            dtype(str): "tensor" or "array"        
        """
        
        assert dtype in ["tensor", "array"]
        self.dtype = torch.Tensor if dtype == "tensor" else np.ndarray
        self.reset()

    def reset(self):
        self.count = 0
        self.array = []

    def __len__(self):
        return len(self.array)
    
    @property
    def item_shape(self):
        if len(self) == 0: return None
        return self.array[0].shape
    
    def update(self, item):
        assert isinstance(item, (torch.Tensor, np.ndarray)), \
            f"expect value of type torch.Tensor, np.ndarray but got {type(item)}"
        if isinstance(item, torch.Tensor):
            item = item.detach()
            if self.dtype == np.ndarray:
                item = item.cpu().numpy()
        
        self.array.append(item)
        self.count += 1
        
    def cat(self, axis=0):
        """
        concatenate all item into a single tensor/array
        """
        if self.dtype == torch.Tensor:
            return torch.cat(self.array, dim=axis)
        return np.concatenate(self.array, axis=axis)