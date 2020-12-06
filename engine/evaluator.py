from contextlib import contextmanager
import time 
import os, sys

import torch
from tqdm import tqdm
import numpy as np

from .base_classes import BaseEvaluator
from .build import EVALUATOR_REGISTRY


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

@EVALUATOR_REGISTRY.register()
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
        imgs_tildes, losses = self.model(batch.imgs)
        losses.pop('total_loss')


        self.monitor.update_metric(imgs_tildes, batch.imgs)
        self.monitor.update_loss(**losses)

    def after_loop(self):
        """
        """
        self.monitor.eval()
