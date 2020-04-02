import numpy as np

import torch
import pickle
import time

#%matplotlib inline
import matplotlib.pyplot as plt
import scipy.sparse

DATASET_NAME = 'SBM_CLUSTER'
dataset = LoadData(DATASET_NAME) # 29s
trainset, valset, testset = dataset.train, dataset.val, dataset.test

print(trainset)