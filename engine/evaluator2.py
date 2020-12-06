from contextlib import contextmanager
import time 
import os, sys

import torch
# from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np

from .evaluator import Evaluator
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
class Evaluator2(Evaluator):
    """
    for metrics which need whole dataset prediction
    """

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
            pred, _ = self.model(
                imgs=batch.imgs
            )

        self.monitor.update_pred(pred, batch.labels)

