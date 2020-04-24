import torch
from data.data import LoadData
import pickle

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def make_onehot_data(data):
    onehot_data_list = []
    n_classes = torch.unique(data[0].dataset[0]['node_label'], dim=0).size(0)
    for i,  one_data in enumerate(data[0].dataset):
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
    dataset[0].node_labels = []
    for i, data in enumerate(dataset[0].dataset):
        dataset[0].dataset[i]['node_label'] = classical_smoothing(w, data['node_label'], n_classes)
        dataset[0].node_labels.append(classical_smoothing(w, data['node_label'], n_classes))
    w_str = str(w).split(".")[1]
    print(f"writing {dataname} w {w_str}")
    with open(f'data/SBMs/{dataname}'+'_'+f'w{w_str}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print("Done")

def classical_label_smoothing(DataSetName, w_list):
    for dataname in DataSetName:
        for w in w_list:
            with open(f'data/SBMs/{dataname}.pkl', 'rb') as f:
                dataset = pickle.load(f)
                n_classes = make_onehot_data(dataset)
                make_label_smoothing(dataname, w, dataset, n_classes)

DataSetName = ['SBM_CLUSTER', 'SBM_PATTERN']
w_list = [0.05, 1, 0.15, 0.2, 0.25]
classical_label_smoothing(DataSetName, w_list)
