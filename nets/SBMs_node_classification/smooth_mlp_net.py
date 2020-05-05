import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss import LabelSmoothingLoss
import dgl

from layers.mlp_readout_layer import MLPReadout, ResnetMLPReadout


class SmoothMLPNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.gated = net_params['gated']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        feat_mlp_modules = [
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            feat_mlp_modules.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            feat_mlp_modules.append(nn.ReLU())
            feat_mlp_modules.append(nn.Dropout(dropout))
        self.feat_mlp = nn.Sequential(*feat_mlp_modules)
        
        if self.gated:
            self.gates = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.readout_mlp = MLPReadout(hidden_dim, n_classes)
        self.MLP_layer2 = ResnetMLPReadout(hidden_dim+n_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, *args, **kwargs):
        g = kwargs['g']
        h = kwargs['h']
        e = kwargs['e']
        delta = kwargs['delta']
        snorm_n = kwargs['snorm_n']
        snorm_e = kwargs['snorm_e']
        label = kwargs['label']

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)

        # MLP
        h = self.feat_mlp(h)
        if self.gated:
            h = torch.sigmoid(self.gates(h)) * h

        # output
        p = self.readout_mlp(h)

        h = torch.cat((h, label.to(torch.float)), dim=1)
        w = self.MLP_layer2(h)
        w = self.sigmoid(w).to(torch.float)
        w = w.data
        w = w.repeat(1, self.n_classes)
        w = torch.clamp(w, min=0, max=delta).to(device=self.device)
        ones = torch.ones(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        max_entropy = torch.Tensor([1 / label.shape[1]]).repeat(label.shape[0], label.shape[1]).to(torch.float).to(device=self.device)
        g_hat = (ones - w) * label.to(torch.float)+ w * max_entropy
        return p, g_hat

        
    def loss(self, pred, label, onehot=False):
        if onehot == True:
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



        
