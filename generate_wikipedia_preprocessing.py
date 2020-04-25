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
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.model_selection import train_test_split
import pandas as pd


def read_edges(data_dir, fname, g_nx):
    with open(os.path.join(data_dir, fname), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            g_nx.add_edge(row[0], row[1])
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

def read_target(data_dir, fname):
    with open(os.path.join(data_dir, fname), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        return [(row[0], row[1]) for row in reader]

def set_graph_label(graph, labels):
    for id, label in labels:
        graph['node_label'][int(id)] = float(label)
        if not isinstance(graph['node_label'], torch.Tensor):
            graph['node_label'] = torch.Tensor(graph['node_label'])

def load_features(features_path):
    """
    Reading the features from disk.
    :param features_path: Location of feature JSON.
    :return features: Feature hash table.
    """
    features = json.load(open(features_path))
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


def reconstruct_feature(features, vocab, graph):
    for k, v in features.items():
        k = int(k)
        for i, t in enumerate(v):
            graph['node_feat'][k][vocab[int(t)]] = 1
        # graph['node_feat'][k] = torch.Tensor(graph['node_feat'][k]).to(torch.long)
        # graph['node_feat'][k] = torch.Tensor([vocab[int(t)] for t in v])
    graph['node_feat'] = torch.Tensor(graph['node_feat']).to(torch.long)


def load_graph(graph_path):
    """
    Reading a NetworkX graph.
    :param graph_path: Path to the edge list.
    :return graph: NetworkX object.
    """
    data = pd.read_csv(graph_path)
    edges = data.values.tolist()
    edges = [[int(edge[0]), int(edge[1])] for edge in edges]
    graph = nx.from_edgelist(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def set_graph(adj, labels, vocab):
    graph = {}
    graph['W'] = adj
    number_of_nodes = len(labels)
    graph['nb_nodes'] = number_of_nodes
    graph['rand_idx'] = ''
    graph['node_feat'] = [[0 for _ in range(len(vocab))] for i in range(number_of_nodes)]
    graph['node_label'] = [0 for i in range(number_of_nodes)]
    set_graph_label(graph, labels)
    return graph

def main(data_name):
    DATA_DIR = f'data/wikipedia/{data_name}/'
    fname1 = f'musae_{data_name}_edges.csv'
    fname2 = f'musae_{data_name}_features.json'
    fname3 = f'musae_{data_name}_target.csv'
    g_nx = nx.Graph()

    graph_datas = []

    adj = torch.Tensor(read_edges(DATA_DIR, fname1, g_nx))
    nodes = g_nx.nodes

    # graph = load_graph(os.path.join(DATA_DIR, fname1))
    # print(graph)
    labels = read_target(DATA_DIR, fname3)
    features = load_features(os.path.join(DATA_DIR, fname2))
    print(features)

    vocab = build_vocab(features)
    graph = set_graph(adj, labels, vocab)
    reconstruct_feature(features, vocab, graph)
    graph_datas.append(graph)
    print(graph)
    features = graph['node_feat']


    emb = embedding(vocab, output_dim=146)
    h = torch.mm(features.to(torch.float), emb.weight)
    print(h)

from torch import nn
def embedding(vocab, output_dim):
    emb = nn.Embedding(len(vocab), output_dim)
    return emb

datas = ['chameleon', 'crocodile', 'squirrel']
for data_name in datas:
    main(data_name)
