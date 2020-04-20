import torch

def make_onehot(trainset, n_class):
    for i, data_list in enumerate(trainset.dataset):
        trainset.dataset[i]['node_label'] = [torch.nn.functional.one_hot(data) for data in data_list['node_label']]
    result_node_labels = []
    for data_list in trainset.node_labels:
        result_node_labels.append([torch.nn.functional.one_hot(data,n_class) for data in data_list])
    trainset.node_labels = result_node_labels