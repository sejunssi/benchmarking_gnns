import numpy as np
import networkx as nx
import torch
import pickle
import time
import os
import sys
import matplotlib.pyplot as plt
import scipy.sparse
import dgl
import networkx as nx


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



class ProgressSmoothing:
    def __init__(self, g_nx):
        self.g_nx = g_nx

    def _get_weight_list(self, a, m, neighbor_list_dict):
        denominator = 0
        weight_list = [0 for _ in range(m+1)]
        for h in range(0, m+1):
            weighting = np.power(a, (m - h))
            num_nodes = len(neighbor_list_dict[h])
            weight_list[h] = weighting
            denominator += weighting * num_nodes
        return weight_list / denominator
    def nei_dict(self, hop_dict):
        neighbor_list_dict = {}  # neighbor_list_dict = {which_hop: [index1, index5, ....]}
        for u, h in hop_dict.items():  # hop_dict = {neighbor_id : which_hop}
            if not h in neighbor_list_dict.keys():
                n_list = [u]  # include self node
                neighbor_list_dict[h] = n_list
            else:
                neighbor_list_dict[h].append(u)
        return neighbor_list_dict
    def get_neigh_smooth_weight(self, v, a, smoothed_labels):
        #         hop_dict = nx.single_source_shortest_path_length(self.g_nx, v)
        hop_dict = nx.single_source_shortest_path_length(self.g_nx, v, 2)
        neighbor_list_dict = self.nei_dict(hop_dict)
        m = np.max(list(neighbor_list_dict.keys()))
        weight_list = self._get_weight_list(a, m, neighbor_list_dict)
        for h in range(0, m+1):
            for u in neighbor_list_dict[h]:
                if h==0:
                    smoothed_labels[v] = smoothed_labels[v] * weight_list[h]
                else:
                    smoothed_labels[v] += smoothed_labels[u] * weight_list[h]
    def smooth_all(self, labels, a):
        smoothed_labels = labels.copy()
        smoothed_labels = smoothed_labels.astype(float)
        for v in list(self.g_nx.nodes):
            self.get_neigh_smooth_weight(v, a, smoothed_labels)
        return smoothed_labels

def generate_smoothing_file(dataname, W_lists, node_label_list, a):
    print(f"Smoothing {dataname} [a]", a)
    train_label = []
    for W, labels in zip(W_lists, node_label_list):
        g_nx = nx.from_numpy_matrix(W)
        ps = ProgressSmoothing(g_nx=g_nx)
        train_label.append(ps.smooth_all(labels, a))
    node_label = train_label
    node_label_list_data = []
    for idx, smoothed_label in enumerate(node_label):
        data[0].dataset[idx]['node_label'] = torch.tensor(smoothed_label)
        node_label_list_data.append(smoothed_label)
    list_tensor_node_label_data = []
    for node_label_data in node_label_list_data:
        list_tensor_node_label_data.append(node_label_data)
    data[0].node_labels = torch.Tensor(list_tensor_node_label_data)
    print("Write")
    with open(f'{dataname}_a{a}.pkl', 'wb') as f:
        pickle.dump(data, f)


import numpy as np


def make_onehot_data(data):
    onehot_data_list = []
    for i,  one_data in enumerate(data[0].dataset):
        n_class = torch.unique(one_data['node_label'], dim=0).size(0)
        node_label = one_data['node_label'].to(torch.int64)
        node_onehot_label = torch.nn.functional.one_hot(node_label, n_class)
        data[0].dataset[i]['node_label'] = node_onehot_label
        onehot_data_list.append(node_onehot_label)
    data[0].node_labels = onehot_data_list


DataSetName = ['SBM_CLUSTER', 'SBM_PATTERN']
for a in [2,3,4,8,1]:
    for dataname in DataSetName:
        with open(f'{dataname}.pkl', 'rb') as f:
            data = pickle.load(f)
        make_onehot_data(data)
        W_lists = list(map(lambda d: d['W'].numpy(), data[0].dataset))
        node_label_list = list(map(lambda d: d['node_label'].numpy(), data[0].dataset))
        generate_smoothing_file(dataname, W_lists, node_label_list,a)
