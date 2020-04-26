import pickle
from data.SBMs import SBMsSampleDatasetDGL

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def read_pickle(dataname, split):
    with open(f'data/SBMs/'+dataname+f"_{split}.pkl", 'rb') as f:
        data = pickle.load(f)
        return data

def load_sample_data(data_name):
    dataset = SBMsSampleDatasetDGL('data/SBMs/', data_name)
    with open('data/SBMs/SBM_CLUSTER_SAMPLE.pkl', 'wb') as f:
        pickle.dump([dataset.train, dataset.val, dataset.test], f)

data_name = "SBM_CLUSTER_SAMPLE"

load_sample_data(data_name)

