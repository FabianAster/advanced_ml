#!/bin/bash

# Define an array of learning rates
learning_rates=(0.1 1 10)

# Define an array of network sizes
train_freq=(1 8 32 128)

# Loop through each learning rate
for lr in "${learning_rates[@]}"; do
  # Loop through each network size
  for freq in "${train_freq[@]}"; do
    python train.py --algo ddpg --env Pendulum-v1 -n 25000 --eval-freq 1000 --save-freq 50 --log-folder ddpg_train --hyperparams gamma:0.98 batch_size:64 learning_rate:$lr buffer_size:1000000 train_freq:$freq tau:0.02 policy_kwargs:"dict(net_arch=[264,264])"
    mv ./ddpg_train/ddpg/Pendulum-v1_1 ./ddpg_train/ddpg/lr${lr}freq${freq}
    python train.py --algo td3 --env Pendulum-v1 -n 25000 --eval-freq 1000 --save-freq 50 --log-folder td3_train --hyperparams gamma:0.98 batch_size:64 learning_rate:$lr buffer_size:1000000 train_freq:$freq tau:0.08 policy_kwargs:"dict(net_arch=[64,64])"
    mv ./td3_train/td3/Pendulum-v1_1 ./td3_train/td3/lr${lr}freq${freq}
    python train.py --algo sac --env Pendulum-v1 -n 25000 --eval-freq 1000 --save-freq 50 --log-folder sac_train --hyperparams gamma:0.98 batch_size:2048 learning_rate:$lr buffer_size:100000 train_freq:$freq tau:0.005 policy_kwargs:"dict(net_arch=[400,300])"
    mv ./sac_train/sac/Pendulum-v1_1 ./sac_train/sac/lr${lr}freq${freq}
  done
done
