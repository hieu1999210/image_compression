import cProfile, pstats, io
import os
import torch

from engine import build_evaluator, build_trainer, build_monitor
from utils import (
    set_deterministic, 
    parse_args, 
    setup_config, 
    get_checkpointer,
    get_log,
    get_cfg_defaults
)
import modelling
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(cfg, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "test":
        # os.makedirs(cfg.DIRS.EXPERIMENT, exist_ok=True)
        # logger = get_log("testing", cfg.DIRS.EXPERIMENT)
        # print(cfg)
        # print(args)
        # model, loss_names = Trainer.build_model(cfg=cfg, device=device, logger=logger)

        # if args.load[-4:] == ".pth":
        #     Checkpointer(
        #         cfg=cfg,
        #         logger=logger, 
        #         model=model,
        #     )._load_state(torch.load(args.load))

        # # tester = Tester2(
        # tester = ClsTester(
        # # tester = PatchTester(
        # # tester = Tester3(
        #     dataloader=Trainer.build_dataloader(cfg, "test", logger),
        #     model=model,
        #     logger=logger,
        #     cfg=cfg,
        #     device=device,
        # )
        # tester.test()
        pass
    elif args.mode == "valid":
        logger = get_log("validation", cfg.DIRS.EXPERIMENT)
        model, loss_names = Trainer.build_model(cfg=cfg, device=device, logger=logger)
        # cfg.DATA.SCALE = 1
        Checkpointer(
            cfg=cfg,
            logger=logger, 
            model=model,
        )._load_state(torch.load(args.load))

        val_dataloader = Trainer.build_dataloader(cfg, "val", logger)
        monitor = get_monitor(
            name=cfg.MODEL.MONITOR_NAME,
            loss_names=loss_names,
            cfg=cfg,
            image_ids=val_dataloader.dataset.image_ids
        )
        evaluator = Evaluator(
            val_dataloader=val_dataloader,
            monitor=monitor,
            model=model,
            logger=logger,
            cfg=cfg,
            device=device,
        )
        evaluator.run_eval()
        print(monitor.map)
        print(monitor.table)
            
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