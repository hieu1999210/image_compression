import argparse
import os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
        help="config yaml path")
    parser.add_argument("--load", type=str, default=None,
        help="whether to resume training")
    parser.add_argument("--resume", action="store_true",
        help="whether to resume training")
    parser.add_argument("-m", "--mode", type=str, default="train",
        help="model runing mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
        help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
        help="enable evaluation mode for testset")
    parser.add_argument("--test_data_dir", type=str, default="",
        help="path to test dataset")
    parser.add_argument("--test_metadata", type=str, default="",
        help="path to test dataset json")
    parser.add_argument("--test_output", type=str, default="",
        help="output path of test set must end with .csv")
    parser.add_argument("--dataset", type=str, default="Training",
        help="dataset for inference 'Validation' or 'Tesing'")
    parser.add_argument("--profiler", action="store_true",
        help="analyse execution time")
    parser.add_argument("--tta", type=str, default="",
        help="")
    parser.add_argument("-d", "--debug", action="store_true",
        help="enable debug mode for test")
    parser.add_argument("--activation", type=str, default="",
        help="activation for infer")
    parser.add_argument("--test_batch_size", type=int, default=1,
        help="batch size for inference")
    parser.add_argument("--save_output", action="store_true",
        help="batch size for inference")
    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def setup_config(cfg, args):
    """
    merge config from yaml file (if any)
    and modify config according to args
    """
    if args.config:
        cfg.merge_from_file(args.config)
    if args.debug:
        opts = ["DEBUG", True,]# "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
        # args.profiler = True
        
    exp_name = os.path.split(args.config)[1].replace(".yaml", "")
    experiment_dir = os.path.join(cfg.DIRS.OUTPUTS, exp_name)
    cfg.merge_from_list([
        'DIRS.EXPERIMENT', experiment_dir,
        'VAL.SAVE_OUTPUT', args.save_output
    ])

    return cfg

