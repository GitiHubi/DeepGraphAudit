#!/bin/bash

# set directory parameter
run_dir=sc_sap_002
base_dir=./200_experiments/$run_dir
data_dir=./100_datasets
log_dir=./300_logs/$run_dir

# set seed parameter
declare -a seeds=(1111 2222 3333 4444 1234)

# set alpha parameter
declare -a betas=(0.0 0.1 0.2)

# set data preparation parameter
dataset=sap
sample_eval=False
sample_size=2001

# set architecture parameter
#declare -a encoder_dim=(128 64 32 16 8 4 2)
#declare -a decoder_dim=(2 4 8 16 32 64 128)
declare -a encoder_dim=(8 4 2)
declare -a decoder_dim=(2 4 8)
declare -a bottlenecks=(linear)
embed_dim=2

# set nn training parameters
batch_size=128  # max regular batch size
iterations=100001  # number of iterations
eval_iteration=5000  # eval iterations

# set learning rate parameter
declare -a rates=(0.01 0.001 0.0001)
# 0.001 0.0001 0.00001

# set anomaly detection parameter
algo=hdbscan
min_cluster_size=100 # the hdbscan min cluster size
min_samples=1 # the hdbscan min samples

# set hardware parameter
device=cuda

# create logs directory
mkdir $log_dir

# iterate over bottlenecks
for bottleneck in "${bottlenecks[@]}"
do

  # iterate over betas
  for beta in "${betas[@]}"
  do

    # iterate over seeds
    for rate in "${rates[@]}"
    do

      # iterate over seeds
      for seed in "${seeds[@]}"
      do

        # determine current timestamp
        timestamp=$(date +%Y-%m-%d_%H-%M-%S)

        # create logs directory
        mkdir $log_dir/$timestamp

        # log experiment progress
        log_timestamp=$(date +%Y-%m-%d_%H-%M-%S)
        echo "[LOG] $log_timestamp : seed $seed, encoder ${encoder_dim[@]}, decoder ${decoder_dim[@]}, rate $rate experiments started"

        # define experiment postfix
        exp_postfix=${run_dir}_bt_${bottleneck}

        # run experiment
        echo "nohup python3 main.py -exp_timestamp $timestamp -base_dir $base_dir -data_dir $data_dir -seed $seed -beta $beta -dataset $dataset -sample_eval $sample_eval -sample_size $sample_size -iterations $iterations -eval_iteration $eval_iteration -batch_size $batch_size -learning_rate $rate -encoder_dim ${encoder_dim[@]} -bottleneck $bottleneck -embed_dim $embed_dim -decoder_dim ${decoder_dim[@]} -device $device -algo $algo -min_cluster_size $min_cluster_size -min_samples $min_samples -exp_postfix ${exp_postfix} > $log_dir/$timestamp/exp_sd_${seed}.log &"
        nohup python3 main.py -exp_timestamp $timestamp -base_dir $base_dir -data_dir $data_dir -seed $seed -beta $beta -dataset $dataset -sample_eval $sample_eval -sample_size $sample_size -iterations $iterations -eval_iteration $eval_iteration -batch_size $batch_size -learning_rate $rate -encoder_dim ${encoder_dim[@]} -bottleneck $bottleneck -embed_dim $embed_dim -decoder_dim ${decoder_dim[@]} -device $device -algo $algo -min_cluster_size $min_cluster_size -min_samples $min_samples -exp_postfix ${exp_postfix} > $log_dir/$timestamp/exp_sd_${seed}.log &

        wait

        # log experiment progress
        log_timestamp=$(date +%Y-%m-%d_%H-%M-%S)
        echo "[LOG] $log_timestamp : seed $seed, encoder ${encoder_dim[@]}, decoder ${decoder_dim[@]}, rate $rate experiments completed"

      done

    done

  done

done
