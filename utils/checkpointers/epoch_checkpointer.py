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

import torch
import os
import json
from collections import deque        


class EpochCheckpointer:
    """
    manage checkpoint
    ### currently assume ema_cp alwa
    working scenarios:
    i. training from scratch: (also include training from public pretraned wieght)
        pretrained weight (locally) and checkpoint are None
        
    ii. finetune: load pretrained weights (local) to model, 
        other checkpointables object are initialized from cfg
        pretrained_w is not None, and there is no existing checkpoint logs
        
    iii. resume: resume training procedure from existing checkpoint log,
        pretrained weight are

    """
    def __init__(self, cfg, logger, model, **checkpointables):
        """
        args: 
            
        """
        self.model = model
        self.checkpointables = checkpointables
        self.logger = logger
        self.save_freq = cfg.SOLVER.SAVE_CHECKPOINT_FREQ
        self.total_epoch_num = cfg.SOLVER.NUM_EPOCHS
        self.num_ckpt = cfg.SOLVER.NUM_CHECKPOINTS

        # make checkpoint dirs
        cp_folder = os.path.join(cfg.DIRS.EXPERIMENT, "checkpoints")
        if not os.path.exists(cp_folder): 
            os.makedirs(cp_folder)
        self.cp_folder = cp_folder
        
        # init cp_logs
        self.cp_logs = self._load_cp_log() # if there is no existing logs, get dict with None values
        
        # store addtional info for each checkpoints
        self.additional_info = {"epoch": 1, "best_metric": -1.0}
        
    def save_checkpoint(self, is_val, **additional_info):
        """
        ensure that keywords of addtional info are different from checkpointables
        
        args:
            -- cp_name (str): checkpoint's file name
            -- additional_info: must have epoch, and 
        """
        # pdate metric

        current_metric = additional_info.pop("current_metric") if is_val else -1.1
        is_best = current_metric > self.additional_info["best_metric"]
        if is_best:
            self.logger.info(
                f"main metric improve from {self.additional_info['best_metric']} to {current_metric}"
            ) 
            self.additional_info["best_metric"] = current_metric
        else:
            self.logger.info(
                f"current best {self.additional_info['best_metric']}"
            ) 
        # update other info
        self.additional_info.update(additional_info)
        
        # save checkpoint
        current_epoch = additional_info["epoch"]
        
        cp = {
            "state_dict": self.model.state_dict(), 
        }
        for key, ob in self.checkpointables.items():
            cp[key] = ob.state_dict()
        cp.update(self.additional_info)
        
        path = os.path.join(self.cp_folder, f"epoch_{current_epoch:0>3}.pth")
        torch.save(cp, path)
        self.logger.info(f"saved new checkpoint to {path}")
        
        # save_best
        cp_name = ""
        if is_best:
            cp_name = f"epoch_{current_epoch:0>3}_{current_metric:.4f}.pth"
            path = os.path.join(self.cp_folder, cp_name)
            torch.save(cp, path)
            self.logger.info(f"saved new best checkpoint to {path}")

        self._update_cp_log(cp_name)

    def load_resume(self, pretrained_w=None):
        """
        load pretrained weights (if any) or resume training
        args: 
            -- pretrained_w (str): path to pretrained weights
        return current epoch, current_best_metric
         
        NOTE: epoch numbering in logging and checkpoint start from 1,
            assume larger metric is better
        """
        # print("##########", pretrained_w)
        # scenario i.
        last_checkpoint = self.cp_logs["last_checkpoint"]
        if  last_checkpoint is not None:
            cp_path = os.path.join(self.cp_folder, last_checkpoint)
            state_dict = self._load_cp_file(cp_path)
            state_dict["epoch"] += 1
            self._load_state(state_dict)
            self.logger.info(f"resume from checkpoint {last_checkpoint}")
            return state_dict["epoch"]
        
        # scenario ii.
        elif pretrained_w is not None:
            weights = self._load_cp_file(pretrained_w)["state_dict"]
            self.model.load_state_dict(weights)
            self.ema_model.load_state_dict(weights)
            self.cp_logs["pretrained"] = pretrained_w
            self.logger.info(f"finetune from pretrained weight: {pretrained_w}")
            return 1
        
        # scenario iii.
        else:
            self.logger.info("##### training from scratch #####")
            return 1

    def _update_cp_log(self, best_cp_name):
        """
        delete least recent checkpoint, update new checkpoint, and save log
        """
        cp_name = f"epoch_{self.additional_info['epoch']:0>3}.pth"
        # get current least_recent checkpoint
        least_recent_ckpt = self.cp_logs["least_recent_checkpoint"]
        # if exist, delete 
        if (
            least_recent_ckpt is not None and 
            len(self._all_cps) >= self.num_ckpt
        ):
            path = os.path.join(self.cp_folder, least_recent_ckpt)
            os.remove(path)

        self._all_cps.append(cp_name)
        self.cp_logs["last_checkpoint"] = cp_name

        least_recent_idx = max(0, len(self._all_cps) - self.num_ckpt)
        self.cp_logs["least_recent_checkpoint"] = self._all_cps[least_recent_idx]

        if best_cp_name:
            current_best = self.cp_logs["best_checkpoint"]
            if current_best:
                os.remove(os.path.join(self.cp_folder, current_best))
            self.cp_logs["best_checkpoint"] = best_cp_name
        path = os.path.join(self.cp_folder, "checkpoints_logs.json")
        
        if len(self._all_cps) > self.num_ckpt:
            self._all_cps.popleft()
        self.cp_logs["all_checkpoints"] = list(self._all_cps)
        with open(path, "w") as f:
            json.dump(self.cp_logs, f, indent=4)

    def _get_last_cp(self):
        """
        return path to last checkpoint
        """
        return self.cp_log["last_checkpoint"]
    
    def _load_cp_file(self, path):
        """
        return checkpoint dict
        """
        return torch.load(path)
    
    def _load_cp_log(self):
        """
        only load at init
        """
        path = os.path.join(self.cp_folder, "checkpoints_logs.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                log = json.load(f)
            self._all_cps = deque(log["all_checkpoints"])
            return log
            
        self.logger.info("There is no existing checkpoint log")
        self._all_cps = deque()
        return {
            "last_checkpoint": None,
            "all_checkpoints": [],
            "best_checkpoint": None,
            "pretrained": None,
            "least_recent_checkpoint": None,
        }
    
    def _load_state(self, state_dict):
        """
        load state to model and checkpointable objects and other info
        """
        self.model.load_state_dict(state_dict.pop("state_dict"))
        for key, ob in self.checkpointables.items():
            ob.load_state_dict(state_dict.pop(key))
            
        for key, info in state_dict.items():
            self.additional_info[key] = info