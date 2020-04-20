
############
# SBM_CLUSTER - 4 RUNS
############

seed0=41
seed1=95
seed2=12
seed3=35
code=main_SBMs_node_classification.py
tmux new -s benchmark_SBMs_node_classification -d
tmux send-keys "source activate benchmark_gnn" C-m


#tmux send-keys "
#python $code --dataset 'SBM_CLUSTER_a2' --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a2.json' &
#python $code --dataset 'SBM_CLUSTER_a3' --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a3.json' &
#python $code --dataset 'SBM_CLUSTER_a4' --gpu_id 2 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a4.json' &
#wait" C-m
#
#tmux send-keys "
#python $code --dataset 'SBM_CLUSTER_a8' --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a8.json' &
#python $code --dataset 'SBM_CLUSTER_a1' --gpu_id 1 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a1.json' &
#wait" C-m



for a in 7
do
  dataset=SBM_CLUSTER_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  tmux send-keys "

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done


for a in 7
do
  dataset=SBM_PATTERN_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done

for a in 8
do
  dataset=SBM_CLUSTER_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done


for a in 8
do
  dataset=SBM_PATTERN_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done


for a in 9 10
do
  dataset=SBM_CLUSTER_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done


for a in 9 7 10
do
  dataset=SBM_PATTERN_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done


for a in 6 4
do
  dataset=SBM_CLUSTER_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done


for a in 6 4
do
  dataset=SBM_PATTERN_a${a}
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 2 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --gpu_id 1 --smooth True --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m
done



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













