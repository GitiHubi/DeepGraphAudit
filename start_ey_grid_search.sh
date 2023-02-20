#!/bin/bash

# set directory parameter
run_dir=sc_ey_001
base_dir=./200_experiments/$run_dir
data_dir=./100_datasets
log_dir=./300_logs/$run_dir

# set data preparation parameter
dataset=ey
sample_eval=False
sample_size=2001

# set seed parameter
declare -a seeds=(1111 2222 3333 4444 1234)

# set alpha parameter
declare -a betas=(0.0 0.1 0.5)

# set architecture parameter
declare -a encoder_dim=(32 8 4 2)
declare -a decoder_dim=(2 4 8 32)
declare -a feat_embed_dims=(2)
declare -a lat_embed_dims=(12 8 4 2)
declare -a bottlenecks=(linear)

# set nn training parameters
train_batch_size=64  # max regular batch size
train_iterations=10001  # number of iterations
valid_iteration=1000 # eval iterations

# set learning rate parameter
declare -a learning_rates=(0.01)
learning_rate_steps=4

# create logs directory
mkdir $log_dir

# iterate over bottlenecks
for lat_embed_dim in "${lat_embed_dims[@]}"
do

  # iterate over betas
  for beta in "${betas[@]}"
  do

    # iterate over seeds
    for learning_rate in "${learning_rates[@]}"
    do

      # iterate over seeds
      for feat_embed_dim in "${feat_embed_dims[@]}"
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
          echo "[LOG] $log_timestamp : beta $beta, rate $learning_rate, feat_embed $feat_embed_dim, seed $seed, encoder ${encoder_dim[@]}, decoder ${decoder_dim[@]} experiments started."

          # define experiment postfix
          exp_postfix=${run_dir}_sd_${seed}_fe_${feat_embed_dim}_beta_${beta}

          # run experiment
          echo "nohup python3 main.py -exp_timestamp $timestamp -base_dir $base_dir -data_dir $data_dir -seed $seed -beta $beta -dataset $dataset -sample_eval $sample_eval -sample_size $sample_size -train_iterations $train_iterations -valid_iteration $valid_iteration -train_batch_size $train_batch_size -learning_rate $learning_rate -learning_rate_steps $learning_rate_steps -encoder_dim ${encoder_dim[@]} -feat_embed_dim $feat_embed_dim -lat_embed_dim $lat_embed_dim -decoder_dim ${decoder_dim[@]} -exp_postfix ${exp_postfix} > $log_dir/$timestamp/exp_sd_${seed}.log &"
          # nohup python3 main.py -exp_timestamp $timestamp -base_dir $base_dir -data_dir $data_dir -seed $seed -beta $beta -dataset $dataset -sample_eval $sample_eval -sample_size $sample_size -train_iterations $train_iterations -valid_iteration $valid_iteration -train_batch_size $train_batch_size -learning_rate $learning_rate -learning_rate_steps $learning_rate_steps -encoder_dim ${encoder_dim[@]} -feat_embed_dim $feat_embed_dim -lat_embed_dim $lat_embed_dim -decoder_dim ${decoder_dim[@]} -exp_postfix ${exp_postfix} > $log_dir/$timestamp/exp_sd_${seed}.log &

          wait

          # log experiment progress
          log_timestamp=$(date +%Y-%m-%d_%H-%M-%S)
          echo "[LOG] $log_timestamp : beta $beta, rate $learning_rate, feat_embed $feat_embed_dim, seed $seed, encoder ${encoder_dim[@]}, decoder ${decoder_dim[@]} experiments completed."

        done

      done

    done

  done

done
