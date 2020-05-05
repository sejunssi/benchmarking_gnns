
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

a_list=(10 9 8 7 6 5)
w_list=(05 1 15 2 25)
# residual a

# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_PATTERN_a${a}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  wait" C-m

done


# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_CLUSTER_a${a}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
done


# None residual a
# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_PATTERN_a${a}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  wait" C-m

done


# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_CLUSTER_a${a}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
done


w_list=1 05
# residual w
# shellcheck disable=SC2068
for w in ${a_list[@]}
do
  dataset=SBM_PATTERN_w${w}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  wait" C-m

done


# shellcheck disable=SC2068
for w in ${w_list[@]}
do
  dataset=SBM_CLUSTER_w${w}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  wait" C-m
done




# None residual
# shellcheck disable=SC2068
for w in ${w_list[@]}
do
  dataset=SBM_PATTERN_w${w}

  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --residual=False --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --gpu_id 1 --residual=False --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --gpu_id 2 --residual=False --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${W}.json' &
  python $code --dataset $dataset --gpu_id 3 --residual=False --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --gpu_id 0 --residual=False --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --gpu_id 1 --residual=False --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --gpu_id 2 --residual=False --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --gpu_id 3 --residual=False --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  wait" C-m

done


# shellcheck disable=SC2068
for w in ${w_list[@]}
do
  dataset=SBM_CLUSTER_w${w}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${W}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 0 --train_soft_target True--seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 1 --train_soft_target True--seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 2 --train_soft_target True--seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False   --gpu_id 3 --train_soft_target True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 0 --train_soft_target True --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 1 --train_soft_target True --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 2 --train_soft_target True --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  python $code --dataset $dataset --residual=False  --gpu_id 3 --train_soft_target True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  wait" C-m
done

############
# SBM_CLUSTER - 4 RUNS
############
dataset=SBM_CLUSTER
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER.json' &
wait" C-m


############
# SBM_PATTERN - 4 RUNS
############


dataset=SBM_PATTERN
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN.json' &
wait" C-m
tmux send-keys "
python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
python $code --dataset $dataset --gpu_id 1 --seed $seed1 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
python $code --dataset $dataset --gpu_id 2 --seed $seed2 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
python $code --dataset $dataset --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN.json' &
wait" C-m

tmux send-keys "tmux kill-session -t benchmark_SBMs_node_classification" C-m















