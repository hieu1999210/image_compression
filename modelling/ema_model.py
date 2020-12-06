from copy import deepcopy
import torch


class EMAModel:
    """ 
    Model Exponential Moving Average
    """
    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict)

    def state_dict(self):
        return self.ema.state_dict()
    
    def __call__(self, x):
        self.ema.eval()
        return self.ema(x)
    
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                # if "running_mean" in k or "running_var" in k:
                #     continue
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                if "running_mean" in k or "running_var" in k:
                    ema_v.copy_(model_v)
                else:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)