from contextlib import contextmanager
import time 
import os, sys

import torch
# from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np

# from structures import Boxes
# from visualization import ImgVisualizer
# from utils import CONSECUTIVE_ID_2_CAT_ID
from .base_classes import BaseEvaluator

# from data_utils import save_as_nii, save_as_npy

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class Evaluator(BaseEvaluator):
    def __init__(
        self, val_dataloader, monitor, model, logger, cfg, device, **kwargs):
        """
        get monitor, data loader, model, 
        """
        
        # fmt: off
        self.val_dataloader     = val_dataloader
        self.monitor            = monitor
        self.model              = model
        self.logger             = logger
        self.device             = device
        self.cfg                = cfg
        # fmt: on

    def run_eval(self):
        """
        """
        self.before_loop()
        # num_stats = len(self.monitor)
        with torch.no_grad(), inference_context(self.model):
            tbar = tqdm(range(len(self.val_dataloader)))
            for i in tbar:
                self.step(i)
                tbar.set_description(f"data time: {self.monitor.data_timer}")
        self.after_loop()

    def before_loop(self):
        """
        reset monitor
        """
        self.logger.info('VALIDATION PHASE:')
        self.val_iter = iter(self.val_dataloader)
        self.monitor.reset()
        
    def step(self, i):
        """
        infer step
        """
        assert not self.model.training
        
        # load data
        start = time.perf_counter()
        batch = next(self.val_iter)
        batch.to(self.device)
        data_time = time.perf_counter() - start
        self.monitor.update_time(data_time)

        # with autocast():
        if True:
            pred, _, _ = self.model(
                imgs=batch.imgs, 
            )

        self.monitor.update_metric(pred, batch.labels)

    def after_loop(self):
        """
        """
        self.monitor.eval()
