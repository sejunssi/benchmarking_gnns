import dgl
from data.data import LoadData

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

DATASET_NAME = 'SBM_PATTERN_w1'
dataset = LoadData(DATASET_NAME)

train, test, val = dataset.train, dataset.test, dataset.val