
# %%


import pickle

# % load_ext
# autoreload
# % autoreload


from data.SBMs_smoothing import SBMsDatasetDGL

from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs_smoothing import SBMsDataset

# %%

DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

# %%

print(len(dataset.train))
print(len(dataset.val))
print(len(dataset.test))

print(dataset.train[0])
print(dataset.val[0])
print(dataset.test[0])

# %%

start = time.time()

with open('SBM_CLUSTER_train_new_0403.pkl', 'wb') as f:
    pickle.dump(dataset.train, f)

with open('SBM_CLUSTER_test_new_0403.pkl', 'wb') as f:
    pickle.dump(dataset.test, f)

with open('SBM_CLUSTER_val_new_0403.pkl', 'wb') as f:
    pickle.dump(dataset.val, f)

with open('SBM_CLUSTER_new.pkl', 'wb') as f:
    pickle.dump([dataset.train, dataset.val, dataset.test], f)

print('Time (sec):', time.time() - start)

# %% md

# Test load function


DATASET_NAME = 'SBM_CLUSTER'
dataset = LoadData(DATASET_NAME)  # 29s
trainset, valset, testset = dataset.train, dataset.val, dataset.test


start = time.time()

batch_size = 10
collate = SBMsDataset.collate
print(SBMsDataset)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate)

print('Time (sec):', time.time() - start)  # 0.002s



