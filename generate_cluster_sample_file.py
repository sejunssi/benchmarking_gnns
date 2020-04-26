import time
import torch
import numpy as np
import pickle
import os
from data.SBMs import SBMsDatasetDGL


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


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

def schuffle(W, c):
    # relabel the vertices at random
    idx = np.random.permutation(W.shape[0])
    # idx2=np.argsort(idx) # for index ordering wrt classes
    W_new = W[idx, :]
    W_new = W_new[:, idx]
    c_new = c[idx]
    return W_new, c_new, idx


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


# configuration
SBM_parameters = {}
SBM_parameters['nb_clusters'] = 6
SBM_parameters['size_min'] = 5
SBM_parameters['size_max'] = 35
SBM_parameters['p'] = 0.55
SBM_parameters['q'] = 0.25
print(SBM_parameters)

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

def SBMs_CLUSTER(nb_graphs, data_dir, name):
    dataset = generate_semisuperclust_dataset(nb_graphs)
    print(len(dataset))
    with open(os.path.join(data_dir, name) + '.pkl', "wb") as f:
        pickle.dump(dataset, f)

start = time.time()
DATA_DIR = 'data/SBMs/'
nb_graphs = 100  # train
# nb_graphs = 3333 # train
# nb_graphs = 500 # train
# nb_graphs = 20 # train
SBMs_CLUSTER(nb_graphs, DATA_DIR, 'SBM_CLUSTER_SAMPLE_train')

nb_graphs = 10  # val
# nb_graphs = 333 # val
# nb_graphs = 100 # val
# nb_graphs = 5 # val
SBMs_CLUSTER(nb_graphs, DATA_DIR, 'SBM_CLUSTER_SAMPLE_val')

nb_graphs = 10  # test
# nb_graphs = 333 # test
# nb_graphs = 100 # test
# nb_graphs = 5 # test
SBMs_CLUSTER(nb_graphs, DATA_DIR, 'SBM_CLUSTER_SAMPLE_test')

print('Time (sec):', time.time() - start)


def load_sample_data(data_name):
    dataset = SBMsDatasetDGL(data_name)
    with open('data/SBMs/SBM_CLUSTER_SAMPLE.pkl', 'wb') as f:
        pickle.dump([dataset.train, dataset.val, dataset.test], f)

data_name = "SBM_CLUSTER_SAMPLE"
load_sample_data(data_name)