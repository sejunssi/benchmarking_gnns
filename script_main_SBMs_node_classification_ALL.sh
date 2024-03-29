
############
# SBM_CLUSTER - 4 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification.py
tmux new -s benchmark_SBMs_node_classification -d
#tmux send-keys "source activate benchmark_gnn" C-m
#
#dataset=SBM_CLUSTER_smoothing
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_smoothing.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_smoothing.json' &
#
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_smoothing.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_smoothing.json' &
#wait" C-m
#
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_smoothing.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_smoothing.json' &
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_smoothing.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_smoothing.json' &
#
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_smoothing.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_smoothing.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_smoothing.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_smoothing.json' &
#wait" C-m


#!/bin/bash

# check :
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_SBMs_node_classification_PATTERN.sh



############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT
#DiffPool





############
# SBM_PATTERN - 4 RUNS
############
dataset=SBM_PATTERN_smoothing
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_smoothing.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_smoothing.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_smoothing.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_smoothing.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_smoothing.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_smoothing.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_smoothing.json' &
wait" C-m

# add last GIN in here
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_smoothing.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN_smoothing.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_smoothing.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_smoothing.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_smoothing.json' &
wait" C-m




#!/bin/bash

# check :
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_SBMs_node_classification_PATTERN.sh

#
#dataset=SBM_CLUSTER
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
#
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
#
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
#
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
#python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
#wait" C-m

############
# GNNs
############

#GatedGCN
#GCN
#GraphSage
#MLP
#GIN
#MoNet
#GAT
#DiffPool





############
# SBM_PATTERN - 4 RUNS
############

dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &

wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
wait" C-m

tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
wait" C-m

tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification" C-m















