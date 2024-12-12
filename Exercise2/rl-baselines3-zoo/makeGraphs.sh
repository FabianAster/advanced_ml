#!/bin/bash

# Define an array of learning rates
learning_rates=(0.00001 0.001 0.00005 0.0001 0.0002)

# Loop through each learning rate
for lr in "${learning_rates[@]}"; do
  python train.py --algo ddpg --env Pendulum-v1 -n 25000 --log-folder ddpg_train --hyperparams gamma:0.98 batch_size:64 learning_rate:$lr buffer_size:1000000 train_freq:256 tau:0.02 policy_kwargs:"dict(net_arch=[256, 256])"
  python train.py --algo td3 --env Pendulum-v1 -n 25000 --log-folder td3_train --hyperparams gamma:0.98 batch_size:64 learning_rate:$lr buffer_size:1000000 train_freq:256 tau:0.08 policy_kwargs:"dict(net_arch=[64, 64])"
  python train.py --algo sac --env Pendulum-v1 -n 25000 --log-folder sac_train --hyperparams gamma:0.98 batch_size:2048 learning_rate:$lr buffer_size:100000 train_freq:256 tau:0.005 policy_kwargs:"dict(net_arch=[400, 300])"
done
