#!/bin/bash

tmux new -s benchmark_SBMs_node_classification_smooth_1 -d
tmux send-keys "source activate benchmark_gnn" C-m


seed1=41
seed2=95
seed3=12
seed4=35

#### CLUSTER residual
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1  --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2  --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m



#### CLUSTER non-residual
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_CLUSTER --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_SMOOTH.json' &
wait" C-m

####### PATTERN residual

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m


####### PATTERN non residual

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2  --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_SMOOTH.json' &
wait" C-m


#####
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0 --ub_delta=0.2 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m

tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.08 --ub_delta=0.12 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk1 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m
tmux send-keys "
python main_SBMs_node_classification_smooth.py --dataset SBM_PATTERN --middle_dim=32 --residual=False --bottleneck=True --lb_delta=0.05 --ub_delta=0.15 --how_residual=rk2 --gpu_id 1 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_SMOOTH.json' &
wait" C-m


tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification_smooth_1" C-m












