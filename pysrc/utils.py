import logging
import collections


def GetLogger(modelName: str):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(modelName)
    return logger


CONFIG = {
    'CACHE_ITEMS': 10000
}

USE_CHECKPOINTS = True 
write_checkpoints = False 
checkpoint_dir = '../data/checkpoints'


def test_counter():
    test = [1, 1, 2, 3, 3, 3, 9, 9, 8, 8]
    c = collections.Counter(test)
    c_sort = sorted(c.keys())
    v = [(item, c[item]) for item in c_sort]
    print(v)


# test_counter()
