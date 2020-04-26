# {'nb_nodes': 117, 'W': tensor([[0, 0, 1,  ..., 0, 1, 0],
#         [0, 0, 1,  ..., 0, 0, 0],
#         [1, 1, 0,  ..., 1, 1, 0],
#         ...,
#         [0, 0, 1,  ..., 0, 1, 0],
#         [1, 0, 1,  ..., 1, 0, 0],
#         [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.int8), 'rand_idx': tensor([ 30,  81,  78,  18, 104,  72,  22,  61,   4,   3,  31,  51,  58,  16,
#          35, 107,  88,   1,  62,  97,  46,  83,  50, 102,  70,  39,  65, 111,
#          43,  85,  21,  15,  82, 112,  41, 114,  68,  42,  13,  93,  29,  73,
#          75,  76,  25,  91,  53,  12,  77, 116,  47, 101,  37,  33,  66,  96,
#          98,  67,  84,  86,   0, 108,  40,  94,  63,  17,  24, 100,  87,  20,
#          28,  38, 106,  48,  79,  56,   2,  14,  10,  99,  45,  55,  90,  80,
#          64,  32,   5,  44,  59, 115,  95,  27,  60, 109, 103,   8,   6,  57,
#          23,  26,   9,  74,  92,  52,  54,  36,  49,  11,  19,   7, 110,  71,
#          34, 105,  89,  69, 113], dtype=torch.int16), 'node_feat': tensor([0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 0, 0],
#        dtype=torch.int16), 'node_label': tensor([[0.9583, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083],
#         [0.0083, 0.0083, 0.0083, 0.9583, 0.0083, 0.0083],...]}

import csv
import json
import networkx as nx

import torch
import os

from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import pickle
from data.wiki import WikiDatasetDGL

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def read_edges(data_dir, fname, g_nx):
    with open(os.path.join(data_dir, fname), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            g_nx.add_edge(int(row[0]), int(row[1]))
        return nx.to_numpy_matrix(g_nx)

# def read_target(data_dir, fname, graph):
#     with open(os.path.join(data_dir, fname), 'r') as f:
#         reader = csv.reader(f, delimiter=',')
#         next(reader)
#         for row in reader:
#             id, target = row[0], row[1]
#             graph['node_label'][int(id)] = float(row[1])
#         if not isinstance(graph['node_label'], torch.Tensor):
#             graph['node_label'] = torch.Tensor(graph['node_label'])

def read_target(data_dir, fname, type='int'):
    with open(os.path.join(data_dir, fname), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        if type == 'int':
            return [(int(row[0]), int(row[1])) for row in reader]
        return [(row[0], row[1]) for row in reader]


def load_features(features_path, type='str'):
    features = json.load(open(features_path))
    if type == 'int':
        features = {int(k): [int(val) for val in v] for k, v in features.items()}
    elif type == 'float':
        features = {float(k): [float(val) for val in v] for k, v in features.items()}
    else:
        features = {str(k): [str(val) for val in v] for k, v in features.items()}
    return features

def build_vocab(features):
    wordset = set()
    for k, v in features.items():
        for vocab in v:
            wordset.add(int(vocab))
    vocab = {word: i+2 for i, word in enumerate(wordset)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    return vocab


def reconstruct_feature(features, labels, vocab, graph):
    sorted_features = []
    for idx, label in labels:
        value = features[idx]
        sorted_features.append(value)

    for idx, (label, feat_val) in enumerate(zip(labels,sorted_features)):
        for f_idx, t in enumerate(feat_val):
            graph['node_feat'][idx][vocab[int(t)]] = 1
        graph['node_label'][idx] = label[1]
    graph['node_feat'] = torch.Tensor(graph['node_feat']).to(torch.long)
    graph['node_label'] = torch.Tensor(graph['node_label']).to(torch.float)



# def load_graph(graph_path):
#     """
#     Reading a NetworkX graph.
#     :param graph_path: Path to the edge list.
#     :return graph: NetworkX object.
#     """
#     data = pd.read_csv(graph_path)
#     edges = data.values.tolist()
#     edges = [[int(edge[0]), int(edge[1])] for edge in edges]
#     graph = nx.from_edgelist(edges)
#     graph.remove_edges_from(nx.selfloop_edges(graph))
#     return graph

def get_troch_data_graph(adj, features, labels, vocab):
    graph = DotDict()
    graph.W = torch.Tensor(adj)
    number_of_nodes = len(labels)
    graph.nb_nodes = number_of_nodes
    graph.rand_idx = ''
    graph.node_feat = [[0 for _ in range(len(vocab))] for i in range(number_of_nodes)]
    graph.node_label = [0 for i in range(number_of_nodes)]
    reconstruct_feature(features, labels, vocab, graph)
    return graph


def generate_subgraph_data(G, features, labels):
    adj = nx.to_numpy_matrix(G)
    sub_nodeset = set(G.nodes)
    sub_features = {k: v for k, v in features.items() if k in set(G.nodes)}
    sub_labels = [labels[idx] for idx in sub_nodeset]
    return adj, sub_features, sub_labels



def feature_embedding(graph, vocab, output_dim):
    reconsted_features = graph['node_feat']
    emb = nn.Embedding(len(vocab), output_dim)
    embedded_feature = torch.mm(reconsted_features.to(torch.float), emb.weight)
    graph['node_feat'] = embedded_feature


def random_split(g_nx, train_test_val_ratio):
    rand_idx = np.random.permutation(len(g_nx.nodes))
    print(f"random shuffle: {len(g_nx)}")
    divided_ratio = [int(x * len(g_nx.nodes)) for x in train_test_val_ratio]
    trainset, valset, testset = \
        rand_idx[divided_ratio[1] + divided_ratio[2]:], \
        rand_idx[:divided_ratio[1]], \
        rand_idx[divided_ratio[1]:divided_ratio[1] + divided_ratio[2]]
    return trainset,valset, testset



def sampling(dataset, numb_of_split_data):
    len_of_data = len(dataset)
    # x = int(len_of_data/numb_of_split_data)
    splited_data_size = int(len_of_data/numb_of_split_data)
    remain_size = len_of_data % numb_of_split_data
    # rand_idx = np.random.permutation(len_of_data)
    data_graphs = []

    sampling_start_idx = 0
    while sampling_start_idx < len_of_data:
        if sampling_start_idx + splited_data_size <= len_of_data:
            data_graphs.append(list(dataset[sampling_start_idx:sampling_start_idx+splited_data_size]))
        elif sampling_start_idx + remain_size <= len_of_data:
            data_graphs[-1].extend(dataset[sampling_start_idx:])
        else:
            break
        sampling_start_idx += splited_data_size
    return data_graphs



def main(data_name):
    DATA_PATH = 'data/wikipedia'
    DATA_DIR = f'{DATA_PATH}/{data_name}/'
    edges_fname= f'musae_{data_name}_edges.csv'
    features_fname = f'musae_{data_name}_features.json'
    target_fname = f'musae_{data_name}_target.csv'
    g_nx = nx.Graph()

    #create graph
    read_edges(DATA_DIR, edges_fname, g_nx)
    g_nx.remove_edges_from(nx.selfloop_edges(g_nx))

    total_labels = read_target(DATA_DIR, target_fname)
    total_features = load_features(os.path.join(DATA_DIR, features_fname), 'int')

    train_test_val_ratio = [10/12, 1/12, 1/12]
    trainset, valset, testset = random_split(g_nx, train_test_val_ratio)

    total_train_SG = g_nx.subgraph(trainset)
    total_test_SG = g_nx.subgraph(testset)
    total_val_SG = g_nx.subgraph(valset)

    train_graphs = sampling(trainset, len(trainset)//100)
    val_graphs = sampling(valset, len(valset)//50)
    test_graphs = sampling(testset, len(testset)//50)

    train_SG = [total_train_SG.subgraph(sg) for sg in train_graphs]
    val_SG = [total_val_SG.subgraph(sg) for sg in val_graphs]
    test_SG= [total_test_SG.subgraph(sg) for sg in test_graphs]


    train_torch_graphs = get_torch_graphs(total_features, total_labels, train_SG)
    val_torch_graphs = get_torch_graphs(total_features, total_labels, val_SG)
    test_torch_graphs = get_torch_graphs(total_features, total_labels, test_SG)
    with open(f'{DATA_DIR}/{data_name}_train.pkl', 'wb') as f:
        print("writing", f'{DATA_DIR}{data_name}_train.pkl')
        pickle.dump(train_torch_graphs, f)
    with open(f'{DATA_DIR}/{data_name}_val.pkl', 'wb') as f:
        print("writing", f'{DATA_DIR}{data_name}_train.pkl')
        pickle.dump(val_torch_graphs, f)
    with open(f'{DATA_DIR}/{data_name}_test.pkl', 'wb') as f:
        print("writing", f'{DATA_DIR}{data_name}_train.pkl')
        pickle.dump(test_torch_graphs, f)
    print("writing", data_name)
    dataset = WikiDatasetDGL(f'data/wikipedia/{data_name}', data_name)
    with open(f'data/wikipedia/{data_name}/{data_name}.pkl', 'wb') as f:
        pickle.dump([dataset.train, dataset.val, dataset.test], f)

def get_torch_graphs(total_features, total_labels, sg):
    torch_graphs = []
    for g in sg:
        adj, feat, label = generate_subgraph_data(g, total_features, total_labels)
        vocab = build_vocab(feat)
        graph = get_troch_data_graph(adj, feat, label, vocab)
        feature_embedding(graph, vocab, 146)
        torch_graphs.append(graph)
    return torch_graphs


datas = ['chameleon', 'crocodile', 'squirrel']
for data_name in datas:
    main(data_name)
