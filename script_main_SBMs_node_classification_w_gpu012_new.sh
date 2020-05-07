#!/bin/bash


code=main_SBMs_node_classification.py
tmux new -s benchmark_SBMs_node_classification_smooth -d
tmux send-keys "source activate benchmark_gnn" C-m


seed_list=(41 95 12 35)
seed1=41
seed2=95
seed3=12
seed4=35

tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w05   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m



tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1   --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15   --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w05   --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1   --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15   --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1   --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15   --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w05   --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1   --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15   --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m



tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1   --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1   --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15   --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15   --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w05   --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05   --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1   --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1   --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15   --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15   --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m


tmux send-keys "
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w05.json' &
python $code --dataset SBM_CLUSTER_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w05.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w1.json' &
python $code --dataset SBM_CLUSTER_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w1.json' &
wait" C-m

tmux send-keys "
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w15.json' &
python $code --dataset SBM_CLUSTER_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed4  --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w15.json' &
wait" C-m


#
#
##P
## shellcheck disable=SC2068
#for seed in ${seed_list[@]}
#do
#
#    tmux send-keys "
#    python $code --dataset SBM_PATTERN_w05   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
#    python $code --dataset SBM_PATTERN_w05   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
#    python $code --dataset SBM_PATTERN_w05   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
#    wait" C-m
#
#    tmux send-keys "
#    python $code --dataset SBM_PATTERN_w1   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
#    python $code --dataset SBM_PATTERN_w1   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
#    python $code --dataset SBM_PATTERN_w1   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
#    wait" C-m
#
#    tmux send-keys "
#    python $code --dataset SBM_PATTERN_w15   --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
#    python $code --dataset SBM_PATTERN_w15   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
#    python $code --dataset SBM_PATTERN_w15   --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
#    wait" C-m
#
#
#    tmux send-keys "
#    python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w05.json' &
#    python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w05.json' &
#    python $code --dataset SBM_PATTERN_w05  --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w05.json' &
#    wait" C-m
#
#    tmux send-keys "
#    python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w1.json' &
#    python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w1.json' &
#    python $code --dataset SBM_PATTERN_w1  --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w1.json' &
#    wait" C-m
#
#    tmux send-keys "
#    python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 0 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w15.json' &
#    python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w15.json' &
#    python $code --dataset SBM_PATTERN_w15 --residual=False --gpu_id 2 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w15.json' &
#    wait" C-m
#
#
#done

tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification_smooth" C-m












