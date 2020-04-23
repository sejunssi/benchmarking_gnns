import torch
from data.data import LoadData
import pickle

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

DATASET_NAME = 'SBM_CLUSTER_SAMPLE'
dataset = LoadData(DATASET_NAME)
w_list = [0.9]

def make_onehot_data(data):
    onehot_data_list = []
    for i,  one_data in enumerate(data[0].dataset):
        n_classes = torch.unique(one_data['node_label'], dim=0).size(0) #TODO fix later
        node_label = one_data['node_label'].to(torch.int64)
        node_onehot_label = torch.nn.functional.one_hot(node_label, n_classes)
        data[0].dataset[i]['node_label'] = node_onehot_label
        onehot_data_list.append(node_onehot_label)
    data[0].node_labels = onehot_data_list
    return n_classes

def classical_smoothing(w, data, n_classes):
    smoothed_label = (1-w) * data.to(torch.float) + torch.ones(data.shape[0], data.shape[1]).to(torch.float)/n_classes * w
    return smoothed_label


def make_label_smoothing(dataname, w, dataset, n_classes):
    dataset[0].dataset = [classical_smoothing(w, data['node_label'], n_classes) for data in dataset[0].dataset]
    w_str = str(w).split(".")[1]
    print(f"writing w {w_str}")
    with open(f'{dataname}'+'_'+f'w{w_str}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print("Done")

def classical_label_smoothing(w_lsit, dataset):
    DataSetName = ['SBM_CLUSTER', 'SBM_PATTERN']
    for dataname in DataSetName:
        with open(f'{dataname}.pkl', 'rb') as f:
            dataset = pickle.load(f)
            for w in w_lsit:
                n_classes = make_onehot_data(dataset)
                make_label_smoothing(dataname, w, dataset, n_classes)

classical_label_smoothing(w_list, dataset)
