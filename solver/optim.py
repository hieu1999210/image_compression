import torch
from timm.optim import RMSpropTF


def make_optimizer(cfg, model):
    """
    Create optimizer with per-layer learning rate and weight decay.
    """
    opt_name = cfg.SOLVER.OPT_NAME
    eps = cfg.SOLVER.EPS

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            # print(key)
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr, eps=eps)
    elif opt_name == "rmsprop":
        optimizer = RMSpropTF(params, lr, alpha=0.9, momentum=cfg.SOLVER.MOMENTUM, eps=eps)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM, 
            nesterov=cfg.SOLVER.USE_NESTEROV)
    return optimizer
