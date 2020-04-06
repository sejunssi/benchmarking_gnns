
import time
import os
import pickle
import numpy as np

import dgl
import torch




# class load_SBMsDataSetDGL(torch.utils.data.Dataset):
#
#     def __init__(self,
#                  data_dir,
#                  name,
#                  split):
#
#         self.split = split
#         self.is_test = split.lower() in ['test', 'val']
#         with open(os.path.join(data_dir, name + '_%s.pkl' % self.split), 'rb') as f:
#             self.dataset = pickle.load(f)
#         self.node_labels = []
#         self.graph_lists = []
#         self.n_samples = len(self.dataset)
#         self._prepare()
#
#
#     def _prepare(self):
#
#         print("preparing %d graphs for the %s set..." % (self.n_samples, self.split.upper()))
#
#         for data in self.dataset:
#
#             node_features = data.node_feat
#             edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list
#
#             # Create the DGL Graph
#             g = dgl.DGLGraph()
#             g.add_nodes(node_features.size(0))
#             g.ndata['feat'] = node_features.long()
#             for src, dst in edge_list:
#                 g.add_edges(src.item(), dst.item())
#
#             # adding edge features for Residual Gated ConvNet
#             #edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
#             edge_feat_dim = 1 # dim same as node feature dim
#             g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim)
#
#             self.graph_lists.append(g)
#             self.node_labels.append(data.node_label)
#
#
#     def __len__(self):
#         """Return the number of graphs in the dataset."""
#         return self.n_samples
#
#     def __getitem__(self, idx):
#         """
#             Get the idx^th sample.
#             Parameters
#             ---------
#             idx : int
#                 The sample index.
#             Returns
#             -------
#             (dgl.DGLGraph, int)
#                 DGLGraph with node feature stored in `feat` field
#                 And its label.
#         """
#         return self.graph_lists[idx], self.node_labels[idx]
#
#
# class SBMsDatasetDGL(torch.utils.data.Dataset):
#
#     def __init__(self, name):
#         """
#             TODO
#         """
#         start = time.time()
#         print("[I] Loading data ...")
#         self.name = name
#         data_dir = 'data/SBMs'
#         self.train = load_SBMsDataSetDGL(data_dir, name, split='train')
#         self.test = load_SBMsDataSetDGL(data_dir, name, split='test')
#         self.val = load_SBMsDataSetDGL(data_dir, name, split='val')
#         print("[I] Finished loading.")
#         print("[I] Data load time: {:.4f}s".format(time.time()-start))
#
#
#
#
# def self_loop(g):
#     """
#         Utility function only, to be used only when necessary as per user self_loop flag
#         : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
#
#
#         This function is called inside a function in SBMsDataset class.
#     """
#     new_g = dgl.DGLGraph()
#     new_g.add_nodes(g.number_of_nodes())
#     new_g.ndata['feat'] = g.ndata['feat']
#
#     src, dst = g.all_edges(order="eid")
#     src = dgl.backend.zerocopy_to_numpy(src)
#     dst = dgl.backend.zerocopy_to_numpy(dst)
#     non_self_edges_idx = src != dst
#     nodes = np.arange(g.number_of_nodes())
#     new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
#     new_g.add_edges(nodes, nodes)
#
#     # This new edata is not used since this function gets called only for GCN, GAT
#     # However, we need this for the generic requirement of ndata and edata
#     new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
#     return new_g
#


class SBMsDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        import networkx as nx
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/SBMs/'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)

            data = f[0]
            W_list = list(map(lambda d: d['W'].numpy(), data))
            W_lists = list(map(lambda d: d['W'].numpy(), data))
            rand_idx_list = list(map(lambda d: d['rand_idx'], data))
            node_feat_list = list(map(lambda d: d['node_feat'], data))
            node_label_list = list(map(lambda d: d['node_label'].numpy(), data))

            class ProgressSmoothing:
                def __init__(self, g_nx):
                    self.g_nx = g_nx

                def _get_weight_list(self, a, m, neighbor_list_dict):
                    denominator = 0
                    weight_list = [0 for _ in range(m)]
                    for h in range(0, m):
                        weighting = np.power(a, (m - h))
                        # print(len(neighbor_list_dict[h]))
                        num_nodes = len(neighbor_list_dict[h])
                        weight_list[h] = weighting * num_nodes

                        #             print(weighting, "@")
                        #             print(num_nodes, "#")
                        denominator += weighting * num_nodes
                    #         print(type(denominator))
                    #         print(type(weight_list))
                    #        print(weight_list/denominator)
                    return weight_list / denominator

                def nei_dict(self, hop_dict):
                    neighbor_list_dict = {}  # neighbor_list_dict = {which_hop: [index1, index5, ....]}
                    for u, h in hop_dict.items():  # hop_dict = {neighbor_id : which_hop}
                        if not h in neighbor_list_dict.keys():
                            n_list = [u]  # include self node
                            neighbor_list_dict[h] = n_list
                        else:
                            neighbor_list_dict[h].append(u)
                    return neighbor_list_dict

                def get_neigh_smooth_weight(self, v, a):
                    #         hop_dict = nx.single_source_shortest_path_length(self.g_nx, v)
                    hop_dict = nx.single_source_shortest_path_length(self.g_nx, v, 2)
                    neighbor_list_dict = self.nei_dict(hop_dict)
                    #         print(neighbor_list_dict)
                    m = np.max(list(neighbor_list_dict.keys()))
                    weight_list = self._get_weight_list(a, m, neighbor_list_dict)
                    # print(weight_list)
                    nidx_weight_list = []
                    for h in range(0, m):
                        for u in neighbor_list_dict[h]:
                            nidx_weight_list.append((int(u), weight_list[h]))
                    return nidx_weight_list

                def smooth_all(self, a, labels):
                    total_nidx_weight_list = []
                    for v in list(g_nx.nodes):
                        # print(v)
                        nidx_weight_list = self.get_neigh_smooth_weight(v, a)
                        # print(nidx_weight_list)
                        total_nidx_weight_list.extend(nidx_weight_list)
                    smoothed_labels = labels.copy()
                    smoothed_labels = smoothed_labels.astype(float)
                    for u, w in total_nidx_weight_list:
                        smoothed_labels[u] *= float(w)
                    return smoothed_labels

            train_label = []
            for W, labels in zip(W_lists, node_label_list):
                # train_W =[]
                #    W = W.numpy()
                #    labels = node_label_list.numpy()
                g_nx = nx.from_numpy_matrix(W)
                ps = ProgressSmoothing(g_nx=g_nx)
                # train_W.append(W)
                train_label.append(ps.smooth_all(2, labels))

            node_label = train_label

            # new_data = [{'W':W, 'rand_idx': rand_idx, 'node_feat': node_feat, 'node_label': node_label}
            #         for W, rand_idx, node_feat, node_label in zip(W_list, rand_idx_list, node_feat_list, node_label)]

            for idx, smoothed_label in enumerate(node_label):
                data[idx]['node_label'] = torch.tensor(smoothed_label)

            # ps = ProgressSmoothing(g_nx=g_nx)
            # smoothed_labels = ps.smooth_all(2, labels)

            # with open('new_SBM_CLUSTER_train_0402_03_dataset.pkl', 'wb') as f:
            #     pickle.dump(data, f)
            #
            # with open('smoothed_SBM_CLUSTER_train_0405', 'wb') as f:
            #     pickle.dump(data, f)

            print('Time (sec):', time.time() - start)

            # f[0] = data

            self.train = data
            self.val = f[1]
            self.test = f[2]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]


SBMsDataset(SBM_CLUSTER)

