"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.wiki_regression.gated_gcn_net import GatedGCNNet
from nets.wiki_regression.gcn_net import GCNNet
from nets.wiki_regression.gat_net import GATNet
from nets.wiki_regression.graphsage_net import GraphSageNet
from nets.wiki_regression.mlp_net import MLPNet
from nets.wiki_regression.gin_net import GINNet
from nets.wiki_regression.mo_net import MoNet as MoNet_

# from nets.wiki_regression.smooth_gcn_net import SmoothGCNNet
# from nets.wiki_regression.smooth_gat_net import SmoothGATNet
# from nets.wiki_regression.smooth_gated_gcn_net import SmoothGatedGCNNet
# from nets.wiki_regression.smooth_graphsage_net import SmoothGraphSageNet
# from nets.wiki_regression.smooth_mo_net import SmoothMoNet
# from nets.wiki_regression.smooth_mlp_net import SmoothMLPNet
# from nets.wiki_regression.smooth_gin_net import SmoothGINNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def GraphSage(net_params):
    return GraphSageNet(net_params)

def MLP(net_params):
    return MLPNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def MoNet(net_params):
    return MoNet_(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        'MLP': MLP,
        'GIN': GIN,
        'MoNet': MoNet,

        # 'SMOOTH_GatedGCN': SmoothGatedGCNNet,
        # 'SMOOTH_GCN': SmoothGCNNet,
        # 'SMOOTH_GAT': SmoothGATNet,
        # 'SMOOTH_GraphSage': SmoothGraphSageNet,
        # 'SMOOTH_MLP': SmoothMLPNet,
        # 'SMOOTH_GIN': SmoothGINNet,
        # 'SMOOTH_Monet': SmoothMoNet,
    }
        
    return models[MODEL_NAME](net_params)