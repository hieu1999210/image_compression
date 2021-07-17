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

import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from bisect import bisect_right


def make_lr_scheduler(cfg, optimizer, num_iters_per_epoch=None):
    """
    NOTE: currently only support iter based training for cosine scheduler
    """

    cfg = cfg.SOLVER
    schedule = cfg.SCHEDULER_NAME

    if cfg.USE_ITER:
        assert schedule in ["cosine_warmup", "constant"]
    num_epochs = cfg.NUM_EPOCHS
    num_decay_epochs = cfg.DECAY_EPOCHS
    num_cosine_cycle = cfg.NUM_COSINE_CYCLE
    decay_rate = cfg.DECAY_RATE
    num_warmup_epochs = cfg.WARMUP_EPOCHS
    grad_acc_steps = cfg.GD_STEPS

    if cfg.USE_ITER:
        num_training_steps = cfg.NUM_ITERS
        num_warmup_steps = cfg.WARMUP_ITERS
    else:
        num_training_steps = (num_epochs * num_iters_per_epoch) // grad_acc_steps
        num_warmup_steps = (num_warmup_epochs * num_iters_per_epoch) // grad_acc_steps


    if schedule == "constant":
        lr_scheduler = get_constant_schedule(optimizer)
    elif schedule == "constant_warmup":
        lr_scheduler = get_constant_schedule_with_warmup(optimizer,
            num_warmup_steps)
    elif schedule == "cosine_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps, num_training_steps, num_cosine_cycle)
    elif schedule == "decay_warmup":
        num_decay_steps = (num_decay_epochs * num_iters_per_epoch) // grad_acc_steps
        lr_scheduler = get_decay_schedule_with_warmup(optimizer, 
            num_training_steps, num_warmup_steps, num_decay_steps, decay_rate)
    elif schedule == "multistep_warmup":
        lr_scheduler = WarmupMultiStepLR(optimizer, 
            cfg.STEPS, cfg.GAMMA, cfg.WARMUP_FACTOR, cfg.WARMUP_ITERS, cfg.WARMUP_METHOD)
    else:
        raise ValueError(f"unknown scheduler: {schedule}")
    return lr_scheduler


def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1, min_lr=0.0):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, math.cos(math.pi*2.0 * float(num_cycles) * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_decay_schedule_with_warmup(optimizer, num_training_steps, num_warmup_steps, num_decay_steps, gamma=0.97, last_epoch=-1):
    """ Create a schedule with learning rate that decays by `gamma`
    every `num_decay_steps`.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        decay_factor = gamma ** math.floor(current_step / num_decay_steps)
        return decay_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]