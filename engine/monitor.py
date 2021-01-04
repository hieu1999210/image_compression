import os
import json
import itertools
import numpy as np
from tabulate import tabulate

from utils import (
    AverageMeter,
    BatchAverageMeter,
    psnr, MS_SSIM
)
from .build import MONITOR_REGISTRY


@MONITOR_REGISTRY.register()
class Monitor:
    """
    NOTE: for metrics which can be calculated accumulatively

    store loss, data time
    and evaluate model in each train/val loop
    Attrbutes:
        data_timer: AverageMeter
        
        loss_a: AverageMeter
        loss_b: AverageMeter
        ...
        
        metric_a: AverageMeter
        metric_b: AverageMeter
        ...

        results: dict: evaluation result at the end of the loop
    """

    def __init__(
        self, 
        loss_names,
        cfg, 
        metric_dict={"psnr": psnr, "ms_ssim": MS_SSIM()},
        logger=None,
        ):
        """
        init storers and averagemeter for the given labels and loss
        
        args:
            metric_dict (list(str)): metric_function 
                e.g. {"iou": iou_fn, "dice_score": dice_score_fn}
                
            -- loss_names (list(str)): names of losses to be stored, 
                e.g. ["loss", "birads", "density"]
        """
        
        # fmt: off
        self.data_timer     = AverageMeter(cache=False)
        self.loss_names     = loss_names
        self.metric_fns     = metric_dict
        self.metric_names   = list(metric_dict.keys())
        self.cfg            = cfg
        self.logger         = logger

        # fmt: on
        for name in loss_names:
            setattr(self, name, AverageMeter(cache=False))
        

        for metric in self.metric_names:
            setattr(self, metric, BatchAverageMeter(cache=False))

        self.reset()

    def reset(self):
        """
        reset all storers and average meter
        """
        for name in self.metric_names:
            getattr(self, name).reset()
        
        for name in self.loss_names:
            getattr(self, name).reset()

        self.data_timer.reset()
        self.results = None

    def update_loss(self, **loss_dict):
        """
        add iter loss to monitor

        args: loss_dict {
            "loss_1": loss_1,
            "loss_2": loss_2,
            ...
        }
        NOTE: keywords must be consistent with loss_names
        """
        for name, value in loss_dict.items():
            getattr(self, name).update(value)

    def update_time(self, data_time):
        """
        update data-time meter
        """
        self.data_timer.update(data_time)
    
    def update_metric(self, preds, targets):
        """
        compute metric for batch of data 
        args: 
            preds: logits
            target
        NOTE: assume pred and target are in range [0,1]
        """
        assert preds.max() <= 1.
        preds = preds*255.
        targets = targets*255.
        for name, fn in self.metric_fns.items():
            getattr(self, name).update(fn(preds, targets))

    def get_mean_loss(self):
        """
        return 
        mean_loss_dict {
            "loss_1": mean_loss_1,
            "loss_2": mean_loss_2,
            ...
        }
        """
        return {f"{name}": getattr(self, name).avg 
                for name in self.loss_names}

    def get_mean_metric(self):
        """
        return 
        mean_metric_dict {
            "metric_1": mean_metric_1,
            "metric_2": mean_metric_2,
            ...
        }
        """
        return {name: getattr(self, name).avg for name in self.metric_names}
    
    def get_metric_val(self):
        """
        """
        return {name: getattr(self, name).val for name in self.metric_names}
    
    def get_loss_val(self):
        """
        """
        return {f"{name}": getattr(self, name).val 
                for name in self.loss_names}
    
    # def get_loss_array(self):
    #     """
    #     return 
    #     loss_array_dict {
    #         "loss_1": array of loss_1,
    #         "loss_2": array of loss_2,
    #         ...
    #     }
    #     """
    #     return {f"{name}_losses": getattr(self, name).array 
    #             for name in self.loss_names}

    def to_str(self, delimiter=","):
        """
        return current value and mean of each loss and data time in string
        """
        str_list = [f"{name+'loss':40s}: {getattr(self, name)}" for name in self.loss_names]
        # str_list.extend([f"{name:40s}: {getattr(self, name)}" for name in self.all_metrics])
        str_list.append(f"{'data time':40s}: {self.data_timer}")
        return f"{delimiter}".join(str_list)
    
    def current_info(self):
        items = [(name, str(getattr(self, name))) for name in self.loss_names]
        items.extend([(name, str(getattr(self, name))) for name in self.metric_names])
        items.append(("Data time", str(self.data_timer)))
        
        return items

    def __len__(self):
        "num_losses + data_time"
        return len(self.loss_names) + len(self.metric_names) + 1

    def eval(self):
        """
        """ ###################################
        self.results = self.get_mean_metric()
        self.results.update(self.get_mean_loss())
