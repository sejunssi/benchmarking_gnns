
############
# SBM_CLUSTER - 4 RUNS  
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification.py 
tmux new -s benchmark_SBMs_node_cluster -d
tmux send-keys "source activate benchmark_gnn" C-m

for a in 1 2 3 4 8
do
  dataset=SBM_CLUSTER_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &

  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &

  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &

  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  wait" C-m


  tmux send-keys "
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &

  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  wait" C-m
done
tmux send-keys "tmux kill-session -t benchmark_SBMs_node_cluster" C-m


#!/bin/bash

# check :
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_SBMs_node_classification_PATTERN.sh













