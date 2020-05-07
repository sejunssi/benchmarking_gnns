
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

w_list=(05 1 15)

# shellcheck disable=SC2068
for seed in ${seed_list[@]}
do
  for w in ${w_list[@]}
  do
    dataset=SBM_CLUSTER_w${w}

    tmux send-keys "
    python $code --dataset $dataset    --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset     --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset    --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
    wait" C-m


    tmux send-keys "
    python $code --dataset $dataset   --residual=False  --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GIN_CLUSTER_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset  --residual=False   --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset  --residual=False  --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GAT_CLUSTER_w${w}.json' &
    wait" C-m



    dataset=SBM_PATTERN_w${w}

    tmux send-keys "
    python $code --dataset $dataset    --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset     --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset    --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset  --residual=False  --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GIN_PATTERN_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset  --residual=False   --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_w${w}.json' &
    wait" C-m

    tmux send-keys "
    python $code --dataset $dataset  --residual=False  --gpu_id 3 --train_soft_target True --seed $seed --config 'configs/SBMs_node_clustering_GAT_PATTERN_w${w}.json' &
    wait" C-m

  done
done