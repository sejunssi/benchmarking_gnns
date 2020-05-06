import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothingLoss

import dgl

import numpy as np

"""
    GMM: Gaussian Mixture Model Convolution layer
    Geometric Deep Learning on Graphs and Manifolds using Mixture Model CNNs (Federico Monti et al., CVPR 2017)
    https://arxiv.org/pdf/1611.08402.pdf
"""

from layers.gmm_layer import GMMLayer
from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout
from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout, RK2netMLPReadout, RK3netMLPReadout

class SmoothMoNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        kernel = net_params['kernel']                       # for MoNet
        dim = net_params['pseudo_dim_MoNet']                # for MoNet
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']                      
        graph_norm = net_params['graph_norm']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']  
        self.device = net_params['device']
        self.n_classes = n_classes
        self.how_residual = net_params['how_residual']
        if self.how_residual == 'resnet':
            self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
        elif self.how_residual == 'rk2':
            self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
        elif self.how_residual == 'rk3':
            self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)
        
        aggr_type = "sum"                                    # default for MoNet
        
        self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.pseudo_proj = nn.ModuleList()

        # Hidden layer
        for _ in range(n_layers-1):
            self.layers.append(GMMLayer(hidden_dim, hidden_dim, dim, kernel, aggr_type,
                                        dropout, graph_norm, batch_norm, residual))
            self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
            
        # Output layer
        self.layers.append(GMMLayer(hidden_dim, out_dim, dim, kernel, aggr_type,
                                    dropout, graph_norm, batch_norm, residual))
        self.pseudo_proj.append(nn.Sequential(nn.Linear(2, dim), nn.Tanh()))
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.MLP_layer2 = ResnetMLPReadout(out_dim + n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *args, **kwargs):
        g = kwargs['g']
        h = kwargs['h']
        e = kwargs['e']
        delta = kwargs['delta']
        snorm_n = kwargs['snorm_n']
        snorm_e = kwargs['snorm_e']
        label = kwargs['label']
        h = self.embedding_h(h)
        
        # computing the 'pseudo' named tensor which depends on node degrees
        us, vs = g.edges()
        # to avoid zero division in case in_degree is 0, we add constant '1' in all node degrees denoting self-loop
        pseudo = [ [1/np.sqrt(g.in_degree(us[i])+1), 1/np.sqrt(g.in_degree(vs[i])+1)] for i in range(g.number_of_edges()) ]
        pseudo = torch.Tensor(pseudo).to(self.device)
        
        for i in range(len(self.layers)):
            h = self.layers[i](g, h, self.pseudo_proj[i](pseudo), snorm_n)

        # output
        p = self.MLP_layer(h)

        h = torch.cat((h, label.to(torch.float)), dim=1)
        w = self.w_layer(h).to(torch.float)
        w = self.sigmoid(w)
        w = w.data
        w = w.repeat(1, self.n_classes)
        w = torch.clamp(w, min=0, max=delta).to(device=self.device)
        ones = torch.ones(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        max_entropy = torch.Tensor([1 / label.shape[1]]).repeat(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        g_hat = (ones - w) * label.to(torch.float) + w * max_entropy
        return p, g_hat
        
    def loss(self, pred, label, train_soft_target):
        if train_soft_target == True:
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
            weight *= (cluster_sizes>0).float()

            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(pred, label)
        return loss