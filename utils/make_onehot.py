import torch

def make_onehot(trainset, n_class):
    result_node_labels = []
    for data_list in trainset.node_labels:
        result_node_labels.append([torch.nn.functional.one_hot(data,n_class) for data in data_list])
    trainset.node_labels = result_node_labels