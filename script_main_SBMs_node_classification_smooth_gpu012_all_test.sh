#!/bin/bash


code1=main_SBMs_node_classification_smooth.py
code2=main_SBMs_node_classification.py
tmux new -s benchmark_SBMs_node_classification_smooth -d
tmux send-keys "source activate benchmark_gnn" C-m


seed_list=(41 95 12 35)

# shellcheck disable=SC2068
#for seed in ${seed_list[@]}
#do
#    tmux send-keys "
#    python $code2 --dataset SBM_PATTERN_w05   --gpu_id 0 --train_soft_target True --seed 41 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
#    python $code2 --dataset SBM_PATTERN_w1    --gpu_id 1 --train_soft_target True --seed 41 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
#    python $code1 --dataset SBM_PATTERN --how_residual=resnet  --gpu_id 2 --seed 41 --config 'configs/SBMs_node_clustering_GIN_PATTERN_SMOOTH.json' &
#    wait" C-m
#
#done

for seed in ${seed_list[@]}
do

    tmux send-keys "
    python main_SBMs_node_classification_smooth --dataset SBM_CLUSTER --delta=0.1 --how_residual=rk2  --gpu_id 0 --seed 41 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json'
    python main_SBMs_node_classification_smooth --dataset SBM_CLUSTER --delta=0.1 --how_residual=rk3  --gpu_id 1 --seed 41 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json'
    python main_SBMs_node_classification_smooth --dataset SBM_CLUSTER --delta=0.2 --how_residual=rk2  --gpu_id 2 --seed 41 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json'
    wait" C-m

    tmux send-keys "
    python main_SBMs_node_classification_smooth --dataset SBM_CLUSTER --delta=0.2 --how_residual=rk2  --gpu_id 1 --seed 41 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_SMOOTH.json'
    wait" C-m
done







tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification_smooth" C-m












