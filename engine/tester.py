import time
import os

import torch
# from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
import pandas as pd

from .my_evaluator import Evaluator, inference_context
from utils import AverageMeter
from data_utils import save_as_nii, save_as_npy

class Tester(Evaluator):
    def __init__(self, dataloader, model, device, logger, cfg):
        
        assert cfg.MODEL.LOSS.ACTIVATION == "softmax", "currently work for softmax only"
        # fmt: off
        self.dataloader     = dataloader
        self.model          = model
        self.logger         = logger
        self.data_timer     = AverageMeter(cache=False)

        self.dataset        = cfg.TEST.DATASET
        self.result_folder  = cfg.TEST.OUTPUT_DIR
        self.device         = device
        # fmt: on

    def test(self):
        
        self.logger.info("INFERING ...")
        self.data_iter = iter(self.dataloader)
        
        with torch.no_grad(), inference_context(self.model):
            tbar = tqdm(range(len(self.dataloader)))
            for _ in tbar:
                self.step()
                tbar.set_description(f"Data time: {self.data_timer}")
        
    def step(self):
        """
        infer step
        """
        assert not self.model.training
        
        # load data
        start = time.perf_counter()
        batch = next(self.data_iter)
        batch.to(self.device)
        data_time = time.perf_counter() - start
        self.data_timer.update(data_time)
        
        # with autocast():
        if True:
            prob, _ = self.model(batch.images)

        # pred[pred==3] = 4
        prob = prob.detach().cpu().numpy()
        for idx, item in enumerate(batch):
            self._save_pred(item[0], item[2], item[3], prob[idx])
    
    def _save_pred(self, study_id, boundary, ori_shape, pred):
        ori_shape = tuple([pred.shape[0]] + list(ori_shape))
        boundary = tuple([slice(pred.shape[0])] + boundary)
        result = -np.ones(ori_shape, dtype=np.float32)
        result[boundary] = pred
        path = os.path.join(
            self.result_folder,
            f"BraTS20_{self.dataset}_{study_id:0>3}_seg.npy"
        )
        save_as_npy(result, path)