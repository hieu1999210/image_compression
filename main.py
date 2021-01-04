import cProfile, pstats, io
import os
import torch

from engine import build_evaluator, build_trainer, build_monitor, Trainer
from utils import (
    set_deterministic, 
    parse_args, 
    setup_config, 
    get_checkpointer,
    get_log,
    get_cfg_defaults
)
import modelling


def main(cfg, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "valid":
        logger = get_log("validation", cfg.DIRS.EXPERIMENT)
        model, loss_names = Trainer.build_model(cfg=cfg, device=device, logger=logger)
        # cfg.DATA.SCALE = 1
        get_checkpointer(cfg.SOLVER.CHECKPOINTER_NAME)(
            cfg=cfg,
            logger=logger, 
            model=model,
        )._load_state(torch.load(args.load))

        val_dataloader = Trainer.build_dataloader(cfg, "val", logger)
        monitor = build_monitor(
            loss_names=loss_names,
            cfg=cfg,
            logger=logger,
        )
        evaluator = build_evaluator(
            val_dataloader=val_dataloader,
            monitor=monitor,
            model=model,
            logger=logger,
            cfg=cfg,
            device=device,
        )
        evaluator.run_eval()
        print(monitor.results)
            
    elif args.mode == "train":
        trainer = build_trainer(cfg=cfg, args=args, device=device)
        trainer.train()
    
    else:
        raise ValueError("Invalid mode")


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    args = parse_args()
    cfg = get_cfg_defaults()
    cfg = setup_config(cfg, args)
    set_deterministic()
    # print(cfg)
    # exit()
    if args.profiler:
        pr = cProfile.Profile()
        pr.enable()
        main(cfg, args)
        pr.disable()
        pr.dump_stats(os.path.join(cfg.DIRS.EXPERIMENT, "log.cprof"))
        s = io.StringIO()
        sort_key = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats()
        print(s.getvalue(), file=open(os.path.join(cfg.DIRS.EXPERIMENT, 'profilers.logs'), 'w'))
    else:
        main(cfg, args)