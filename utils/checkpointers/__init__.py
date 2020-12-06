from .iter_checkpointer import IterCheckpointer
from .epoch_checkpointer import EpochCheckpointer


def get_checkpointer(name):
    # name = cfg.SOlVER.CHECKPOINTER_NAME
    return {
        "IterCheckpointer": IterCheckpointer,
        "EpochCheckpointer": EpochCheckpointer,
    }[name] 