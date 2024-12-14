#!/bin/bash

# Define an array of learning rates
learning_rates=(0.00001 0.001 0.00005 0.0001 0.0002)

# Define an array of network sizes
network_sizes=("64,64" "128,128" "256,256")

# Loop through each learning rate
for lr in "${learning_rates[@]}"; do
  # Loop through each network size
  for net_size in "${network_sizes[@]}"; do
    python train.py --algo ddpg --env Pendulum-v1 -n 25000 --log-folder ddpg_train --hyperparams gamma:0.98 batch_size:512 learning_rate:$lr buffer_size:100000 train_freq:32 tau:0.05 policy_kwargs:"dict(net_arch=[$net_size])"
    python train.py --algo td3 --env Pendulum-v1 -n 25000 --log-folder td3_train --hyperparams gamma:0.98 batch_size:64 learning_rate:$lr buffer_size:1000000 train_freq:256 tau:0.08 policy_kwargs:"dict(net_arch=[$net_size])"
    python train.py --algo sac --env Pendulum-v1 -n 25000 --log-folder sac_train --hyperparams gamma:0.98 batch_size:2048 learning_rate:$lr buffer_size:100000 train_freq:256 tau:0.005 policy_kwargs:"dict(net_arch=[$net_size])"
  done
done
