import torch

def make_onehot_node_label(trainset, n_class):
    result_node_labels=[]
    for i, data_list in enumerate(trainset.dataset):
        trainset.dataset[i]['node_label'] = [torch.nn.functional.one_hot(data_list['node_label'].to(torch.int64), n_class)]
        result_node_labels.append(trainset.dataset[i]['node_label'])
    trainset.node_labels = result_node_labels