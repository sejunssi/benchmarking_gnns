import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothingLoss

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout
from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout, RK2netMLPReadout, RK3netMLPReadout
from layers.mlp_readout_layer \
    import MLPReadout, ResnetMLPReadout, RK2netMLPReadout, RK3netMLPReadout, RK2M1netMLPReadout, RKinetMLPReadout, BaseLineMLPReadout

class SmoothGraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.how_residual = net_params['how_residual']

        bottleneck = net_params['bottleneck']
        self.bottleneck = bottleneck
        self.how_residual = net_params['how_residual']
        self.middle_dim = net_params['middle_dim']
        self.rki = net_params['rki']
        self.rk_middle_layer(bottleneck, hidden_dim, n_classes, net_params)
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([GraphSageLayer(hidden_dim, hidden_dim, F.relu,
                                              dropout, aggregator_type, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        self.MLP_layer2 = ResnetMLPReadout(out_dim + n_classes, 1)
        self.sigmoid = nn.Sigmoid()
        

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
        h = self.in_feat_dropout(h)

        # graphsage
        for conv in self.layers:
            h = conv(g, h, snorm_n)

        # output
        p = self.MLP_layer(h)

        h = torch.cat((h, label.to(torch.float)), dim=1)
        if self.middle_dim != 'None':
            h = self.dense_layer(h)
            if self.bottleneck:
                h = self.widen_layer(h)
            h = self.fc_layer(h)
        w = self.w_layer(h).to(torch.float)

        w = self.sigmoid(w).to(torch.float)
        w = w.data
        saved_w = w
        w = w.repeat(1, self.n_classes)
        w = torch.clamp(w, min=lb_delta, max=ub_delta).to(device=self.device)
        ones = torch.ones(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        max_entropy = torch.Tensor([1 / label.shape[1]]).repeat(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        g_hat = (ones - w) * label.to(torch.float)+ w * max_entropy
        return  p, g_hat, g, saved_w

    def rk_middle_layer(self, bottleneck, hidden_dim, n_classes, net_params):
        if self.middle_dim != 'None':
            self.middle_dim = net_params['middle_dim']
            middle_dim = self.middle_dim
            self.dense_layer = nn.Linear(hidden_dim + n_classes, middle_dim)
            if bottleneck == True:
                self.widen_layer = nn.Linear(middle_dim, hidden_dim + n_classes)
                self.fc_layer = nn.Linear(hidden_dim + n_classes, hidden_dim + n_classes)
                if self.how_residual == 'rki':
                    self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
                elif self.how_residual == 'rk2m1':
                    self.w_layer = RK2M1netMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk2':
                    self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk1':
                    self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
                elif self.how_residual == 'rk3':
                    self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)
                else:
                    self.w_layer = BaseLineMLPReadout(hidden_dim + n_classes, 1)
            else:
                self.fc_layer = nn.Linear(middle_dim, middle_dim)
                if self.how_residual == 'rki':
                    self.w_layer = RKinetMLPReadout(middle_dim, 1, self.rki)
                elif self.how_residual == 'rk2m1':
                    self.w_layer = RK2M1netMLPReadout(middle_dim, 1)
                elif self.how_residual == 'rk2':
                    self.w_layer = RK2netMLPReadout(middle_dim, 1)
                elif self.how_residual == 'rk1':
                    self.w_layer = ResnetMLPReadout(middle_dim, 1)
                elif self.how_residual == 'rk3':
                    self.w_layer = RK3netMLPReadout(middle_dim, 1)
                else:
                    self.w_layer = BaseLineMLPReadout(middle_dim, 1)
        else:
            if self.how_residual == 'rki':
                self.w_layer = RKinetMLPReadout(hidden_dim + n_classes, 1, self.rki)
            elif self.how_residual == 'rk2m1':
                self.w_layer = RK2M1netMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk2':
                self.w_layer = RK2netMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk1':
                self.w_layer = ResnetMLPReadout(hidden_dim + n_classes, 1)
            elif self.how_residual == 'rk3':
                self.w_layer = RK3netMLPReadout(hidden_dim + n_classes, 1)
            else:
                self.w_layer = BaseLineMLPReadout(hidden_dim + n_classes, 1)

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
            weight *= (cluster_sizes>0).float()

            # weighted cross-entropy for unbalanced classes
            criterion = nn.CrossEntropyLoss(weight=weight)
            loss = criterion(pred, label)

        return loss


        
