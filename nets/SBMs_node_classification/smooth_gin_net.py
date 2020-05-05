import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothingLoss

import dgl
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""

from layers.gin_layer import GINLayer, ApplyNodeFunc, MLP

class SmoothGINNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        self.n_layers = net_params['L']
        n_mlp_layers = net_params['n_mlp_GIN']               # GIN
        learn_eps = net_params['learn_eps_GIN']              # GIN
        neighbor_aggr_type = net_params['neighbor_aggr_GIN'] # GIN
        readout = net_params['readout']                      # this is graph_pooling_type
        graph_norm = net_params['graph_norm']      
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        
        self.embedding_h = nn.Embedding(in_dim, hidden_dim)
        
        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction1 = torch.nn.ModuleList()
        self.linears_prediction2 = torch.nn.ModuleList()

        self.conv = self.conv1x1(hidden_dim + n_classes, hidden_dim)
        self.norm_layer = nn.BatchNorm2d(hidden_dim)
        for layer in range(self.n_layers+1):
            self.linears_prediction1.append(nn.Linear(hidden_dim, n_classes))


            self.linears_prediction2.append(nn.Linear(hidden_dim, 1))

        self.sigmoid = nn.Sigmoid()

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        
    def forward(self, *args, **kwargs):
        g = kwargs['g']
        h = kwargs['h']
        e = kwargs['e']
        delta = kwargs['delta']
        snorm_n = kwargs['snorm_n']
        snorm_e = kwargs['snorm_e']
        label = kwargs['label']
        h = self.embedding_h(h)
        
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.n_layers):
            h = self.ginlayers[i](g, h, snorm_n)
            hidden_rep.append(h)

        score_over_layer_p = 0
        score_over_layer_w = 0

        for i, h in enumerate(hidden_rep):
            score_over_layer_p += self.linears_prediction1[i](h)
            x = h
            x = torch.cat((x, label.to(torch.float)), dim=1)
            y = self.conv(x.reshape(x.shape[0], x.shape[1], 1, 1))
            y = self.norm_layer(y).squeeze()
            y = y + h
            score_over_layer_w += self.linears_prediction2[i](y)

        w = self.sigmoid(score_over_layer_w).to(torch.float)
        w = w.data
        w = w.repeat(1, self.n_classes)
        w = torch.clamp(w, min=0, max=delta).to(device=self.device)
        ones = torch.ones(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        max_entropy = torch.Tensor([1 / label.shape[1]]).repeat(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        g_hat = (ones - w) * label.to(torch.float) + w * max_entropy
        return score_over_layer_p, g_hat
        
    def loss(self, pred, label, train_soft_target=False):
        if train_soft_target == True:
            criterion = LabelSmoothingLoss()
            loss = criterion(pred, label)
            return loss
        else:
            # calculating label weights for weighted loss computation
            V = label.size(0)
            label_count = torch.bincount(label)
            # label_unique = label.unique(sorted=True)
            # label_count = torch.stack((label==l_u).sum() for l_u in label_unique)

            label_count = label_count[label_count.nonzero()].squeeze()
            cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
            cluster_sizes[torch.unique(label)] = label_count
            weight = (V - cluster_sizes).float() / V
            weight *= (cluster_sizes>0).float()

            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(pred, label)

        return loss