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
    def smooth_all(self, labels):
        smoothed_labels = labels.copy()
        smoothed_labels = smoothed_labels.astype(float)
        a = len(labels)
        for v in list(self.g_nx.nodes):
            self.get_neigh_smooth_weight(v, a, smoothed_labels)
        return smoothed_labels



def generate_smoothing_file(dataname, W_lists, node_label_list):
    train_label = []
    for W, labels in zip(W_lists, node_label_list):
        g_nx = nx.from_numpy_matrix(W)
        ps = ProgressSmoothing(g_nx=g_nx)
        train_label.append(ps.smooth_all(labels))
    node_label = train_label
    for idx, smoothed_label in enumerate(node_label):
        data[0].dataset[idx]['node_label'] = torch.tensor(smoothed_label)
    data[0].node_labels = []
    for train_dataset in data[0].dataset:
        data[0].node_labels.extend(train_dataset)
    print("Write")
    with open(f'{dataname}_Smoothing.pkl', 'wb') as f:
        pickle.dump(data, f)


DataSetName = ['SBM_PATTERN']
for dataname in DataSetName:
    with open(f'{dataname}.pkl', 'rb') as f:
        data = pickle.load(f)
    W_lists = list(map(lambda d: d['W'].numpy(), data[0].dataset))
    node_label_list = list(map(lambda d: d['node_label'].numpy(), data[0].dataset))
    generate_smoothing_file(dataname, W_lists, node_label_list)
