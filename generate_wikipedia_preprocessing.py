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


def read_edges(data_dir, fname, G, g_nx):
    with open(os.path.join(data_dir, fname), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            G.add_edge(row[0], row[1])
        return nx.to_numpy_matrix(G)

# def read_json(data_dir, fname):
#     with open(fname, 'r') as f:
#         data = json.load(dat)


def read_target(data_dir, fname, graph):
    with open(os.path.join(data_dir, fname), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            id, target = row[0], row[1]
            graph['node_label'][int(id)] = float(row[1])
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
    return vocab


def reconstruct_feature(features, vocab, graph):
    for k, v in features.items():
        k = int(k)
        graph['node_feat'][k] = torch.Tensor([vocab[int(t)] for t in v])

def main():
    DATA_DIR = 'data/wikipedia/chameleon/'
    fname1 = 'musae_chameleon_edges.csv'
    fname2 = 'musae_chameleon_features.json'
    fname3 = 'musae_chameleon_target.csv'
    G = nx.MultiGraph()
    g_nx = nx.Graph()

    graph_datas = []
    graph = {}
    adj = torch.Tensor(read_edges(DATA_DIR, fname1, G, g_nx))
    number_of_nodes = len(G)
    graph['W'] = adj
    graph['nb_nodes'] = len(G)
    graph['rand_idx'] = ''
    graph['node_feat'] = [[] for i in range(len(G))]
    graph['node_label'] = [0 for i in range(len(G))]
    read_target(DATA_DIR, fname3, graph)
    print(graph)

    # def reconstruct_vocab(graph):
    #     word_set = set()
    #     graph['node_feat']
    #     word_set.add()


    features = load_features(os.path.join(DATA_DIR, fname2))
    print(features)


    vocab = build_vocab(features)
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1
    reconstruct_feature(features, vocab, graph)

    print(graph)

from torch import nn
def embedding(features, vocab, output_dim):
    emb = nn.Embedding(len(vocab), output_dim)
    return emb

#
#
# document_collections = create_documents(features)
#
#         model = Doc2Vec(document_collections,
#                         vector_size=self.args.dimensions,
#                         window=0,
#                         min_count=self.args.min_count,
#                         alpha=self.args.alpha,
#                         dm=0,
#                         negative=self.args.negative_samples,
#                         ns_exponent=self.args.exponent,
#                         min_alpha=self.args.min_alpha,
#                         sample=self.args.down_sampling,
#                         workers=self.args.workers,
#                         epochs=self.args.epochs)


# import pandas as pd
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
# graph2 = load_graph(os.path.join(DATA_DIR, fname1))

# def create_documents(features):
#     """
#     From a feature hash create a list of TaggedDocuments.
#     :param features: Feature hash table - keys are nodes, values are feature lists.
#     :return docs: Tagged Documents list.
#     """
#     docs = [TaggedDocument(words=v, tags=[str(k)]) for k, v in features.items()]
#     return docs

# docs = create_documents(features)
# model = Doc2Vec(docs, vector_size=142, window=0)
# print(model)