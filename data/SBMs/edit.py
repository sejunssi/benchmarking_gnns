# %%

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


class load_SBMsDataSetDGL(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 name,
                 split):

        self.split = split
        self.is_test = split.lower() in ['test', 'val']
        with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
            self.dataset = pickle.load(f)
        self.node_labels = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()

    def _prepare(self):

        print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))

        for data in self.dataset:

            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(node_features.size(0))
            g.ndata['feat'] = node_features.long()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())

            # adding edge features for Residual Gated ConvNet
            # edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1  # dim same as node feature dim
            g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)

            self.graph_lists.append(g)
            self.node_labels.append(data.node_label)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        return self.graph_lists[idx], self.node_labels[idx]


class SBMsDatasetDGL(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            TODO
        """
        start = time.time()
        print("[I] Loading data ...")
        self.name = name
        data_dir = 'data/SBMs'
        self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
        self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
        self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))


def self_loop(g):

    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g


class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/SBMs/'
        with open(data_dir + name + '.pkl', "rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]


print(os.getcwd())


# Generate SBM CLUSTER graphs

def schuffle(W, c):
    # relabel the vertices at random
    idx = np.random.permutation(W.shape[0])
    # idx2=np.argsort(idx) # for index ordering wrt classes
    W_new = W[idx, :]
    W_new = W_new[:, idx]
    c_new = c[idx]
    return W_new, c_new, idx


def block_model(c, p, q):
    n = len(c)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if c[i] == c[j]:
                prob = p
            else:
                prob = q
            if np.random.binomial(1, prob) == 1:
                W[i, j] = 1
                W[j, i] = 1
    return W


def unbalanced_block_model(nb_of_clust, clust_size_min, clust_size_max, p, q):
    c = []
    for r in range(nb_of_clust):
        if clust_size_max == clust_size_min:
            clust_size_r = clust_size_max
        else:
            clust_size_r = np.random.randint(clust_size_min, clust_size_max, size=1)[0]
        val_r = np.repeat(r, clust_size_r, axis=0)
        c.append(val_r)
    c = np.concatenate(c)
    W = block_model(c, p, q)
    return W, c


class generate_SBM_graph():

    def __init__(self, SBM_parameters):
        # parameters
        nb_of_clust = SBM_parameters['nb_clusters']
        clust_size_min = SBM_parameters['size_min']
        clust_size_max = SBM_parameters['size_max']
        p = SBM_parameters['p']
        q = SBM_parameters['q']

        # block model
        W, c = unbalanced_block_model(nb_of_clust, clust_size_min, clust_size_max, p, q)

        # shuffle
        W, c, idx = schuffle(W, c)

        # signal on block model
        u = np.zeros(c.shape[0])
        for r in range(nb_of_clust):
            cluster = np.where(c == r)[0]
            s = cluster[np.random.randint(cluster.shape[0])]
            u[s] = r + 1

        # target
        target = c

        # convert to pytorch
        W = torch.from_numpy(W)
        W = W.to(torch.int8)
        idx = torch.from_numpy(idx)
        idx = idx.to(torch.int16)
        u = torch.from_numpy(u)
        u = u.to(torch.int16)
        target = torch.from_numpy(target)
        target = target.to(torch.int16)

        # attributes
        self.nb_nodes = W.size(0)
        self.W = W
        self.rand_idx = idx
        self.node_feat = u
        self.node_label = target


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


def generate_semisuperclust_dataset(nb_graphs):
    dataset = []
    for i in range(nb_graphs):
        if not i % 250:
            print(i)
        data = generate_SBM_graph(SBM_parameters)
        graph = DotDict()
        graph.nb_nodes = data.nb_nodes
        graph.W = data.W
        graph.rand_idx = data.rand_idx
        graph.node_feat = data.node_feat
        graph.node_label = data.node_label
        dataset.append(graph)
    return dataset


def SBMs_CLUSTER(nb_graphs, name):
    dataset = generate_semisuperclust_dataset(nb_graphs)
    print(len(dataset))
    with open(name + '.pkl', "wb") as f:
        pickle.dump(dataset, f)
    # plot_histo_graphs(dataset, name)




from data.SBMs import SBMsDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)

    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['COLLAB', 'ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS:
        return SBMsDataset(DATASET_NAME)

    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for the CITATIONGRAPHS Datasets
    CITATIONGRAPHS_DATASETS = ['CORA', 'CITESEER', 'PUBMED']
    if DATASET_NAME in CITATIONGRAPHS_DATASETS:
        return CitationGraphsDataset(DATASET_NAME)









with open('_SBM_CLUSTER_a1.pkl', 'rb') as f:
    data = pickle.load(f)


with open('SBM_CLUSTER_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_CLUSTER_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_CLUSTER_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)


from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset


DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_CLUSTER_a1.pkl', 'wb') as f:
    pickle.dump(dataset, f)





with open('_SBM_CLUSTER_a2.pkl', 'rb') as f:
    data = pickle.load(f)


with open('SBM_CLUSTER_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_CLUSTER_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_CLUSTER_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)


from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset


DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_CLUSTER_a2.pkl', 'wb') as f:
    pickle.dump(dataset, f)




with open('_SBM_CLUSTER_a3.pkl', 'rb') as f:
    data = pickle.load(f)


with open('SBM_CLUSTER_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_CLUSTER_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_CLUSTER_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)


from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset


DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_CLUSTER_a3.pkl', 'wb') as f:
    pickle.dump(dataset, f)



with open('_SBM_CLUSTER_a4.pkl', 'rb') as f:
    data = pickle.load(f)


with open('SBM_CLUSTER_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_CLUSTER_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_CLUSTER_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)


from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset


DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_CLUSTER_a4.pkl', 'wb') as f:
    pickle.dump(dataset, f)






with open('_SBM_CLUSTER_a8.pkl', 'rb') as f:
    data = pickle.load(f)

with open('SBM_CLUSTER_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_CLUSTER_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_CLUSTER_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)


from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset


DATASET_NAME = 'SBM_CLUSTER'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_CLUSTER_a8.pkl', 'wb') as f:
    pickle.dump(dataset, f)






with open('_SBM_PATTERN_a1.pkl', 'rb') as f:
    data = pickle.load(f)


with open('SBM_PATTERN_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_PATTERN_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_PATTERN_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)


from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset


DATASET_NAME = 'SBM_PATTERN'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_PATTERN_a1.pkl', 'wb') as f:
    pickle.dump(dataset, f)










with open('_SBM_PATTERN_a2.pkl', 'rb') as f:
    data = pickle.load(f)

with open('SBM_PATTERN_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_PATTERN_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_PATTERN_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)

from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset

DATASET_NAME = 'SBM_PATTERN'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_PATTERN_a2.pkl', 'wb') as f:
    pickle.dump(dataset, f)






with open('_SBM_PATTERN_a3.pkl', 'rb') as f:
    data = pickle.load(f)

with open('SBM_PATTERN_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_PATTERN_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_PATTERN_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)

from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset

DATASET_NAME = 'SBM_PATTERN'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_PATTERN_a3.pkl', 'wb') as f:
    pickle.dump(dataset, f)









with open('_SBM_PATTERN_a4.pkl', 'rb') as f:
    data = pickle.load(f)

with open('SBM_PATTERN_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_PATTERN_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_PATTERN_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)

from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset

DATASET_NAME = 'SBM_PATTERN'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_PATTERN_a4.pkl', 'wb') as f:
    pickle.dump(dataset, f)





with open('_SBM_PATTERN_a8.pkl', 'rb') as f:
    data = pickle.load(f)

with open('SBM_PATTERN_train.pkl', 'wb') as f:
    pickle.dump(data[0].dataset, f)

with open('SBM_PATTERN_val.pkl', 'wb') as f:
    pickle.dump(data[1].dataset, f)

with open('SBM_PATTERN_test.pkl', 'wb') as f:
    pickle.dump(data[2].dataset, f)

from data.SBMs import SBMsDatasetDGL
from data.data import LoadData
from torch.utils.data import DataLoader
from data.SBMs import SBMsDataset

DATASET_NAME = 'SBM_PATTERN'
dataset = SBMsDatasetDGL(DATASET_NAME)  # 3983s

with open('SBM_PATTERN_a8.pkl', 'wb') as f:
    pickle.dump(dataset, f)