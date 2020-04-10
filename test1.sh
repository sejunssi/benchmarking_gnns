seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification.py

tmux send-keys "source activate benchmark_gnn" C-m


dataset=SBM_CLUSTER_a1





dataset=SBM_CLUSTER_a2

python main_SBMs_node_classification.py --dataset SBM_CLUSTER_a2 --gpu_id 1 --seed 41 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json'
