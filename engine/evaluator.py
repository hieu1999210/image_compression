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

from contextlib import contextmanager
import time 
import os

import torch
from tqdm import tqdm
import numpy as np
import cv2

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
        self.save_output        = cfg.VAL.SAVE_OUTPUT
        self.cfg                = cfg
        # fmt: on
        
        if self.save_output:
            output_dir = os.path.join(cfg.DIRS.EXPERIMENT, "pred")
            os.makedirs(output_dir)
            self.output_dir = output_dir

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
        img_tildes, losses = self.model(batch.imgs)
        losses.pop('total_loss')

        self.monitor.update_metric(img_tildes, batch.imgs)
        self.monitor.update_loss(**losses)
        if self.save_output:
            self.save_outputs(batch.image_ids, batch.imgs, img_tildes)
            
    def after_loop(self):
        """
        """
        self.monitor.eval()

    def save_outputs(self, ids, x, x_tilde):
        MARGIN = 10
        N,C,H,W = x.size()
        x = np.flip((x*255.).permute(0,2,3,1).cpu().numpy().astype(np.uint8), axis=3)
        x_tilde = np.flip((x_tilde*255.).permute(0,2,3,1).cpu().numpy().astype(np.uint8), axis=3)
        
        img = np.zeros((H,2*W+MARGIN,C),dtype=np.uint8)
        for _x, _x_tilde, idx in zip(x, x_tilde, ids):
            img[:,:W,:] = _x
            img[:,W+MARGIN:,:] = _x_tilde
            path = os.path.join(self.output_dir, f"{idx}.png")
            cv2.imwrite(path, img)
            