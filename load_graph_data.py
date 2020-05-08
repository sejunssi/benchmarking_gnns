import pickle

data_dir = 'result/weight/'
dataname = '41_True_SBM_CLUSTER_SMOOTH_GIN_baseline_rk1_lb0_ub2_md32_True_07-May-2020-16-35.pkl'
with open(data_dir+"/"+dataname, 'rb') as f:
    loaded_data = pickle.load(f)

graph = loaded_data['g']
edge_list = [g.edges() for g in graph]
label = loaded_data['label']
smoothed_label = loaded_data['smoothed_label']
label = loaded_data['label']

# graph_dict = {
#             'g': batch_graph_list,
#             'label': batch_label_list,
#             'smoothed_label': smoothed_label_list,
#             'predicted_label': batch_scores_list
#         }