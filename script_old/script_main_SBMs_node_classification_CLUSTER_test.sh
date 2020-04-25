#!/bin/bash

# check : 
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_SBMs_node_classification_CLUSTER.sh




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
# SBM_CLUSTER - 4 RUNS  
############
#
#seed0=41
#seed1=95
#seed2=12
#seed3=35
#code=main_SBMs_node_classification.py
#tmux new -s benchmark_SBMs_node_classification -d
#tmux send-keys "source activate benchmark_gnn" C-m
#dataset=SBM_CLUSTER
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json'
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json'
#wait" C-m
#tmux send-keys "
#python $code --dataset $dataset --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json'
#wait" C-m
#tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification" C-m





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
  printf $dataset
  tmux send-keys "

  wait" C-m
done
tmux send-keys "tmux kill-session -t benchmark_SBMs_node_cluster" C-m


#!/bin/bash
#python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
# check :
# bash script.sh
# tmux attach -t script_mol_opt
# tmux detach
# pkill python

# bash script_main_SBMs_node_classification_PATTERN.sh


















