import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothingLoss

import dgl
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPReadout


class Smooth_GCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)  # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim, F.relu, dropout,
                                              self.graph_norm, self.batch_norm, self.residual) for _ in
                                     range(n_layers - 1)])
        self.layers.append(
            GCNLayer(hidden_dim, out_dim, F.relu, dropout, self.graph_norm, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

    def forward(self, *args, **kwargs):
        # g = kwargs['g']
        h = kwargs['h']
        # e = kwargs['e']
        delta = kwargs['delta']
        snorm_n = kwargs['snorm_n']
        # snorm_e = kwargs['snorm_e']
        label = kwargs['label']

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # concated_a = torch.cat((h.reshape(len(h), -1), label), dim=1)
        # input embedding
        # h2 = self.embedding_h(h)
        # h2 = self.in_feat_dropout(h2)
        h2 = torch.cat((h, label.to(torch.FloatTensor)), dim=1)

        # GCN1
        for conv in self.layers:
            h1 = conv(g, h, snorm_n)

        #GCN2
        for conv in self.layers:
            h2 = conv(g, h2, snorm_n)

        # output
        p = self.MLP_layer(h1)
        w = self.MLP_layer(h2)
        w = torch.clmap(w, min=0, max=delta)
        g_hat = (torch.ones(label.size)-w) * label + w * torch.Tensor([1/ label.size for r in range(label.size)])
        return p, g_hat

    def loss(self, pred, label, smooth=False):
        if smooth == True:
            criterion = LabelSmoothingLoss()
            loss = criterion(pred, label)
            return loss
        else:
            # calculating label weights for weighted loss computation
            V = label.size(0)
            label_count = torch.bincount(label)
            label_count = label_count[label_count.nonzero()].squeeze()
            cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
            cluster_sizes[torch.unique(label)] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes > 0).float()

            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(pred, label)
        return loss
