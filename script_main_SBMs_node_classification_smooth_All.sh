
############
# SBM_CLUSTER - 4 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification_smooth.py
tmux new -s benchmark_SBMs_node_classification_smooth -d
tmux send-keys "source activate benchmark_gnn" C-m

# residual

############
# SBM_CLUSTER - 4 RUNS
############
dataset=SBM_CLUSTER
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
wait" C-m


############
# SBM_PATTERN - 4 RUNS
############


dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification_smooth" C-m


# No residual

############
# SBM_CLUSTER - 4 RUNS
############
dataset=SBM_CLUSTER
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset ---residual=False -gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_SMOOTH.json' &
wait" C-m


############
# SBM_PATTERN - 4 RUNS
############


dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification_smooth" C-m












