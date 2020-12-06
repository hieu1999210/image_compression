from yacs.config import CfgNode as CN

INF = 1e10

_C = CN()
_C.EXPERIMENT = "test"
_C.SEED = 0
_C.DEBUG = False

_C.DIRS = CN()
_C.DIRS.OUTPUTS = ""
_C.DIRS.EXPERIMENT = ""

### for backward compatible
_C.DIRS.TRAIN_DATA = ""
_C.DIRS.TRAIN_METADATA = ""
_C.DIRS.VAL_DATA = ""

### for semisup
_C.DIRS.DATA = ""

_C.DATA = CN()

# for semisup
_C.DATA.IN_CHANNELS = 3
_C.DATA.NUM_WORKERS = 4
_C.DATA.INTERPOLATE = "bilinear"
_C.DATA.SIZE = 256
# _C.DATA.MAX_SIZE = 1333
_C.DATA.MEAN = [0.,0.,0.]
_C.DATA.STD = [1., 1., 1.]
_C.DATA.TRAIN_DATASET_NAME = ""
_C.DATA.VAL_DATASET_NAME = ""
_C.DATA.DATALOADER_NAME = "infinite_dataloader"

_C.DATA.AUG = CN()


_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "Compressor2018"


### config for analysis and synthesis blocks
_C.MODEL.STRIDES = [2,2,2,2]
_C.MODEL.CONV_KERNEL = 5
_C.MODEL.INTER_CHANNELS = 192
_C.MODEL.LATENT_CHANNELS = 192 #320

### config for prior analysis and prior synthesis blocks
_C.MODEL.HYPER_PRIOR = CN()
_C.MODEL.HYPER_PRIOR.STRIDES = [1,2,2]
_C.MODEL.HYPER_PRIOR.KERNELS = [3,5,5] # for analysis block, need to reverse for synthesis block
### config for entropy model
_C.MODEL.ENTROPY_MODEL = CN()
_C.MODEL.ENTROPY_MODEL.DIMS = [3,3,3]
_C.MODEL.ENTROPY_MODEL.INIT_SCALE = 10
_C.MODEL.ENTROPY_MODEL.BIN = 1.
_C.MODEL.ENTROPY_MODEL.PROB_EPS = 1e-10
_C.MODEL.ENTROPY_MODEL.CONDITIONAL_MODEL = "LaplacianConditionalModel"

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.DISTORTION_LOSS_WEIGHT = 1.

_C.SOLVER = CN()
_C.SOLVER.USE_ITER = True
_C.SOLVER.GD_STEPS = 1 
_C.SOLVER.IMS_PER_BATCH = 2
_C.SOLVER.NUM_CHECKPOINTS = 10
_C.SOLVER.CHECKPOINTER_NAME = "IterCheckpointer"
_C.SOLVER.TRAINER_NAME = "Trainer"
_C.SOLVER.EVALUATOR_NAME = ""
_C.SOLVER.MONITOR_NAME = ""
_C.SOLVER.MAIN_METRIC = ""
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.NUM_ITERS = 1 << 20

# scheduler
_C.SOLVER.SCHEDULER_NAME = "cosine_warmup"
_C.SOLVER.DECAY_EPOCHS = 2.4
_C.SOLVER.DECAY_RATE = 0.97

_C.SOLVER.WARMUP_EPOCHS = 1
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3

_C.SOLVER.WEIGHT_SCHEDULER_NAME = "linear_warmup"
_C.SOLVER.WEIGHT_WARMUP_ITERS = 1

_C.SOLVER.EPS = 1e-4

# for multi step scheduler
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

# optimizer
_C.SOLVER.OPT_NAME = "sgd"
_C.SOLVER.GRAD_CLIP = 0.0
_C.SOLVER.USE_NESTEROV = False
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.NUM_COSINE_CYCLE = 0.21875 # 7/32
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 1. # 2. for sgd



_C.VAL = CN()
_C.VAL.BATCH_SIZE = 1
_C.VAL.SAVE_PRED = False
_C.VAL.ITER_FREQ = 1 << 10
# _C.VAL.USE_EMA = False



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`