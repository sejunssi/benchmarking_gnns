
from data.data import LoadData


DATASET_NAME = 'SBM_CLUSTER_SAMPLE'
dataset = LoadData(DATASET_NAME)

train, test, val = dataset.train, dataset.test, dataset.val