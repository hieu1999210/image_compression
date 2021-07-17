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

from .base_classes import *
from utils import Registry
TRAINER_REGISTRY = Registry("TRAINER")
EVALUATOR_REGISTRY = Registry("EVALUATOR")
MONITOR_REGISTRY = Registry("MONITOR")


def build_trainer(cfg, **kwargs):
    trainer_name = cfg.SOLVER.TRAINER_NAME
    trainer = TRAINER_REGISTRY.get(trainer_name)(cfg=cfg, **kwargs)
    assert isinstance(trainer, BaseTrainer)

    return trainer


def build_evaluator(cfg, **kwargs):
    evaluator_name = cfg.SOLVER.EVALUATOR_NAME
    evaluator = EVALUATOR_REGISTRY.get(evaluator_name)(cfg=cfg, **kwargs)
    assert isinstance(evaluator, BaseEvaluator)

    return evaluator


def build_monitor(cfg, **kwargs):
    monitor_name = cfg.SOLVER.MONITOR_NAME
    monitor = MONITOR_REGISTRY.get(monitor_name)(cfg=cfg, **kwargs)

    return monitor