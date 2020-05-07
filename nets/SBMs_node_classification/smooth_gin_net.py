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
from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout, RK2netMLPReadout, RK3netMLPReadout, \
    RK2M1netMLPReadout, RKinetMLPReadout, BaseLineMLPReadout

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
        bottleneck = net_params['bottleneck']
        self.bottleneck = bottleneck
        self.how_residual = net_params['how_residual']
        self.middle_dim = net_params['middle_dim']
        self.rki = net_params['rki']

        
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()

        self.embedding_h = nn.Embedding(in_dim, hidden_dim)

        for layer in range(self.n_layers):
            mlp = MLP(n_mlp_layers, hidden_dim, hidden_dim, hidden_dim)
            
            self.ginlayers.append(GINLayer(ApplyNodeFunc(mlp), neighbor_aggr_type,
                                           dropout, graph_norm, batch_norm, residual, 0, learn_eps))

        self.rk_middle_layer(bottleneck, hidden_dim, n_classes, net_params)
        # Linear function for output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction1 = torch.nn.ModuleList()

        for layer in range(self.n_layers+1):
            self.linears_prediction1.append(nn.Linear(hidden_dim, n_classes))

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
                elif self.how_residual == 'rk2m1':
                    self.w_layer = RK2M1netMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk2':
                    self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'resnet':
                    self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk3':
                    self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)
                else:
                    self.w_layer = BaseLineMLPReadout(hidden_dim + n_classes, 1)
            else:
                if self.how_residual == 'rki':
                    self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
                elif self.how_residual == 'rk2m1':
                    self.w_layer = RK2M1netMLPReadout(middle_dim + n_classes, 1)
                elif self.how_residual == 'rk2':
                    self.w_layer = RK2netMLPReadout(middle_dim + n_classes, 1)
                elif self.how_residual == 'resnet':
                    self.w_layer = ResnetMLPReadout(middle_dim + n_classes, 1)
                elif self.how_residual == 'rk3':
                    self.w_layer = RK3netMLPReadout(middle_dim + n_classes, 1)
                else:
                    self.w_layer = BaseLineMLPReadout(hidden_dim + n_classes, 1)
        else:
            if self.how_residual == 'rki':
                self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
            elif self.how_residual == 'rk2m1':
                self.w_layer = RK2M1netMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk2':
                self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'resnet':
                self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk3':
                self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)
            else:
                self.w_layer = BaseLineMLPReadout(hidden_dim + n_classes, 1)

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        
    def forward(self, *args, **kwargs):
        g = kwargs['g']
        h = kwargs['h']
        e = kwargs['e']
        lb_delta = kwargs['lb_delta']
        ub_delta = kwargs['ub_delta']
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
            if self.middle_dim != 'None':
                h = self.dense_layer(torch.cat((h, label.to(torch.float)), dim=1))
                if self.bottleneck == True:
                    h = self.widen_layer(h)
            else:
                h = torch.cat((h, label.to(torch.float)), dim=1)
            score_over_layer_w += self.w_layer(h)

        w = self.sigmoid(score_over_layer_w).to(torch.float)
        w = w.data
        saved_w = w
        w = w.repeat(1, self.n_classes)
        w = torch.clamp(w, min=lb_delta, max=ub_delta).to(device=self.device)
        ones = torch.ones(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        max_entropy = torch.Tensor([1 / label.shape[1]]).repeat(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        g_hat = (ones - w) * label.to(torch.float) + w * max_entropy
        return score_over_layer_p, g_hat, g, saved_w
        
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