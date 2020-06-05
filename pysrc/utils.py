import logging


def GetLogger(modelName: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(modelName)
    return logger


use_checkpoints = True
write_checkpoints = False
checkpoint_dir = '../data/checkpoints'
