
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

a_list=(8 7 6)
w_list=(05 1 15)
# residual a


# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_CLUSTER_a${a}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
done

# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_PATTERN_a${a}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  wait" C-m

done

# Non residual
# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_CLUSTER_a${a}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_a${a}.json' &
  wait" C-m
done


# shellcheck disable=SC2068
for a in ${a_list[@]}
do
  dataset=SBM_PATTERN_a${a}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_PATTERN_a${a}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_a${a}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_a${a}.json' &
  wait" C-m

done


# shellcheck disable=SC2068
for w in ${w_list[@]}
do
  dataset=SBM_CLUSTER_w${w}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  wait" C-m
done


# residual w
# shellcheck disable=SC2068
for w in ${a_list[@]}
do
  dataset=SBM_PATTERN_w${w}

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  wait" C-m

done


# shellcheck disable=SC2068
for w in ${w_list[@]}
do
  dataset=SBM_CLUSTER_w${w}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3 --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
  wait" C-m
done





# None residual
# shellcheck disable=SC2068
for w in ${w_list[@]}
do
  dataset=SBM_PATTERN_w${w}

  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3  --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_CLUSTER_w${w}.json' &
  wait" C-m
  tmux send-keys "
  python $code --dataset $dataset --residual=False --gpu_id 3  --seed $seed3 --config 'configs/SBMs_node_clustering_MLP_GATED_CLUSTER_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GraphSage_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False   --gpu_id 3 --onehot True--seed $seed3 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_MoNet_PATTERN_w${w}.json' &
  wait" C-m

  tmux send-keys "
  python $code --dataset $dataset --residual=False  --gpu_id 3 --onehot True --seed $seed3 --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
  wait" C-m

done
