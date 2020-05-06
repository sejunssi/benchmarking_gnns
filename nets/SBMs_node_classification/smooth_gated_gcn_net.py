import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothingLoss

import dgl
import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout, RK2netMLPReadout, RK3netMLPReadout, RK2M1netMLPReadout, RKinetMLPReadout

class SmoothGatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        in_dim_edge = 1 # edge_dim (feat is a float)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.graph_norm = net_params['graph_norm']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']

        bottleneck = net_params['bottleneck']
        self.bottleneck = bottleneck
        self.how_residual = net_params['how_residual']
        self.middle_dim = net_params['middle_dim']
        self.rki = net_params['rki']

        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim) # edge feat is a float
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                       self.graph_norm, self.batch_norm, self.residual) for _ in range(n_layers) ])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)
        self.rk_middle_layer(bottleneck, hidden_dim, n_classes, net_params)
        self.sigmoid = nn.Sigmoid()
    

    def rk_middle_layer(self, bottleneck, hidden_dim, n_classes, net_params):
        if self.middle_dim != 'None':
            self.middle_dim = net_params['middle_dim']
            middle_dim = self.middle_dim
            self.dense_layer = nn.Linear(hidden_dim + n_classes, middle_dim)
            if bottleneck == True:
                self.widen_layer = nn.Linear(middle_dim, hidden_dim + n_classes)
                if self.how_residual == 'rki':
                    self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
                elif self.how_residual == 'rk2_m1':
                    self.w_layer = RK2M1netMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk2':
                    self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'resnet':
                    self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk3':
                    self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)
            else:
                if self.how_residual == 'rki':
                    self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
                elif self.how_residual == 'rk2_m1':
                    self.w_layer = RK2M1netMLPReadout(middle_dim + n_classes, 1)
                elif self.how_residual == 'rk2':
                    self.w_layer = RK2netMLPReadout(middle_dim + n_classes, 1)
                elif self.how_residual == 'resnet':
                    self.w_layer = ResnetMLPReadout(middle_dim + n_classes, 1)
                elif self.how_residual == 'rk3':
                    self.w_layer = RK3netMLPReadout(middle_dim + n_classes, 1)
        else:
            if self.how_residual == 'rki':
                self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
            elif self.how_residual == 'rk2_m1':
                self.w_layer = RK2M1netMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk2':
                self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'resnet':
                self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk3':
                self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)

    def forward(self, *args, **kwargs):
        g = kwargs['g']
        h = kwargs['h']
        e = kwargs['e']
        lb_delta = kwargs['lb_delta']
        ub_delta = kwargs['ub_delta']
        snorm_n = kwargs['snorm_n']
        snorm_e = kwargs['snorm_e']
        label = kwargs['label']

        # input embedding
        h = self.embedding_h(h)
        e = self.embedding_e(e)
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e, snorm_n, snorm_e)

        # output
        p = self.MLP_layer(h)
        # output
        h = torch.cat((h, label.to(torch.float)), dim=1)
        if self.middle_dim != 'None':
            h = self.dense_layer(h)
            if self.bottleneck:
                h = self.widen_layer(h)
        w = self.w_layer(h).to(torch.float)
        w = self.sigmoid(w)
        w = w.data
        saved_w = w
        w = w.repeat(1, self.n_classes).to(device=self.device)
        w = torch.clamp(w, min=lb_delta, max=ub_delta).to(device=self.device)
        ones = torch.ones(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        max_entropy = torch.Tensor([1 / label.shape[1]]).repeat(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        g_hat = (ones - w) * label.to(torch.float) + w * max_entropy
        return p, g_hat, g, saved_w
        

    def loss(self, pred, label, train_soft_target=False):
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
            weight *= (cluster_sizes > 0).float()

            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(pred, label)
        return loss
