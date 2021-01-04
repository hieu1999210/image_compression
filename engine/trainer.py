import time
import os
import csv
import logging
import sys
from tabulate import tabulate

import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
import torch

from solver import make_lr_scheduler, make_optimizer
from data_utils import build_dataloader
from modelling import build_model
from .base_classes import BaseTrainer
from utils import (
    get_log, write_log, AverageMeter, get_checkpointer,
)
from .build import build_monitor, TRAINER_REGISTRY, build_evaluator


@TRAINER_REGISTRY.register()
class Trainer(BaseTrainer):
    """
    """
    def __init__(self, cfg, args, device):
        assert cfg.SOLVER.CHECKPOINTER_NAME == "IterCheckpointer"
        
        # check if overive previous experiments
        if not (args.resume or args.test):
            assert not os.path.isdir(cfg.DIRS.EXPERIMENT),\
                "{} exists".format(cfg.DIRS.EXPERIMENT)
            os.makedirs(cfg.DIRS.EXPERIMENT)
        
        self.cfg = cfg
        self.device = device
        self.iters_per_epoch = cfg.VAL.ITER_FREQ
        self.grad_clip = cfg.SOLVER.GRAD_CLIP
        
        # init global logger
        logger = get_log("main", cfg.DIRS.EXPERIMENT)
        self.logger = logger
        logger.info(f"=========> {cfg.EXPERIMENT} <=========")
        logger.info(f'\n\nStart with config {cfg}')
        logger.info(f'Command arguments {args}')
        
        # init dataloader
        train_dataloader = self.build_dataloader(cfg, "train", logger)
        val_dataloader = self.build_dataloader(cfg, "val", logger)
        self.train_dataloader = train_dataloader
        
        # init model
        model, loss_names = self.build_model(cfg=cfg, logger=logger, device=device)
        logger.info(model)

        # init optimizer
        optimizer = make_optimizer(cfg, model)

        self.model = model
        self.optimizer = optimizer
        
        # init scheduler
        scheduler = make_lr_scheduler(cfg, optimizer)
        self.scheduler = scheduler
        
        # tensorboard
        tb = SummaryWriter(os.path.join(self.cfg.DIRS.EXPERIMENT, "tb"))
        self.tensorboard = tb
        
        # init monitor
        monitor = build_monitor(
            cfg=cfg,
            loss_names=loss_names,
            logger=logger,
        )
        self.monitor = monitor
        
        # init checkpointer
        self.checkpointer = get_checkpointer(cfg.SOLVER.CHECKPOINTER_NAME)(
            cfg=cfg,
            logger=logger, 
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.current_iter = self.checkpointer.load_resume(args.load)

        # init evaluator
        self.evaluator = build_evaluator(
            val_dataloader=val_dataloader,
            monitor=monitor,
            model=model,
            logger=logger,
            cfg=cfg,
            device=self.device,
        )

        # init epoch time meter
        self.epoch_timer = AverageMeter(cache=False)
    
    def train(self):
        num_iters = self.cfg.SOLVER.NUM_ITERS
        end_epoch = time.perf_counter()
        self.model.train()
        
        self.traindata_iter = iter(self.train_dataloader)
        while self.current_iter <= num_iters:

            self.begin_epoch()
            self.train_loop()
            
            # 'un-update' current_iter for logging
            self.current_iter -= 1

            self.after_train_loop()
            
            results = self.validate()
            
            self.end_epoch(results=results, is_val=True)
            self.epoch_timer.update(time.perf_counter() - end_epoch)
            end_epoch = time.perf_counter()
            self.logger.info(
                f'iter: [{(self.current_iter):0>3}/{num_iters:0>3}] \t'
                f'Overall Time {self.epoch_timer} \t'
            )

            self.current_iter += 1
    
    def begin_epoch(self):
        """
        also begin train loop
        """
        self.logger.info(
            f'\n\n*****\ITER {self.current_iter:0>8}/{self.cfg.SOLVER.NUM_ITERS:0>8}:')
        self.logger.info('TRAIN PHASE:')
        self.monitor.reset()
    
    def train_loop(self):
        tbar = tqdm(range(self.iters_per_epoch))
        # tbar = tqdm(range(5))
        num_stats = len(self.monitor) + 2 + 1 # +2 for table border, +lr for progress bar
        for i in tbar:
            self.train_step()
            if i:
                sys.stdout.write("\033[F\033[K"*num_stats) 
            stats = (self.monitor.current_info() + 
                [("Learning rate", f"{self.optimizer.param_groups[0]['lr']:.6f}")])
            tbar.write(tabulate(stats))
            
    def train_step(self):
        """
            i: iter index
            NOTE: currently not eval for training data
        """
        assert self.model.training
        
        # load data
        start = time.perf_counter()
        batch = next(self.traindata_iter)
        batch.to(self.device)
        data_time = time.perf_counter() - start
        self.monitor.update_time(data_time)

        with torch.set_grad_enabled(True):
            # forward
            imgs_tilde, losses = self.model(batch.imgs)
            loss = losses.pop('total_loss').mean()

            # backward
            loss /= self.cfg.SOLVER.GD_STEPS
            if torch.isnan(loss).any():
                print("get nan loss")
                exit()
                
            loss.backward()
            if self.grad_clip:
                clip_grad_value_(self.model.parameters(), self.grad_clip)
            if self.current_iter % self.cfg.SOLVER.GD_STEPS == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()


        # logging
        with torch.no_grad():
            # print(losses)
            self.monitor.update_metric(imgs_tilde, batch.imgs)
            self.monitor.update_loss(**losses)
            log = self.monitor.get_loss_val()
            log.update(self.monitor.get_metric_val())
            log["lr"] = self.optimizer.param_groups[0]["lr"]
            self.tensorboard.add_scalars(
                "train",
                log,
                self.current_iter   
            )
        self.current_iter += 1
    
    def after_train_loop(self):
        """
        evaluating training loop and write out logs
        """
        self.monitor.eval()
        self._save_results(mode="train")
    
    def validate(self):
        self.evaluator.run_eval()
        results = self._save_results(mode="val")
        return results
    
    def end_epoch(self, results=None, is_val=True):
        """
        """
        self.checkpointer.save_checkpoint(
            is_val, 
            iter=self.current_iter, 
            current_metric=-1 if results is None else results[self.cfg.SOLVER.MAIN_METRIC]
        )

    def _save_results(self, mode):
        """
        write results to logging
        args: 
            mode: "val" or "train"
        return results
        """

        assert mode in ["val", "train"], "invalid mode"
        results = self.monitor.results
            
        self.tensorboard.add_scalars(f"{mode}_mean", results, self.current_iter)
        write_log(self.logger.info, mode=mode, **results)

        return results
    
    @classmethod
    def build_model(cls, cfg, device, logger=None):
        if logger:
            logger.info('loading model')
        model = build_model(cfg)
        loss_names = model.loss_names
        
        
        # if torch.cuda.device_count() > 1:
        if True:
            model = torch.nn.DataParallel(model)
        model = model.to(device)
        
        return model, loss_names
        # return model
    
    @classmethod
    def build_dataloader(cls, cfg, mode, logger=None):
        assert mode in ["test", "val", "train"]
        if logger:
            logger.info(f'loading {mode} dataset')
        
        if mode == "test":
            dataloader = get_testdataset(cfg)
        else:
            dataloader = build_dataloader(cfg=cfg, mode=mode)
        
        return dataloader
