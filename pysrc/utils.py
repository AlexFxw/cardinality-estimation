import logging


def GetLogger(modelName: str):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(modelName)
    return logger
