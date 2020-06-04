import logging


def GetLogger(modelName: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(modelName)
    return logger


CONFIG = {
    'CACHE_ITEMS': 1000
}

USE_CHECKPOINTS = True
write_checkpoints = False
checkpoint_dir = '../data/checkpoints'
