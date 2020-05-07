
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