python train.py --algo ddpg --env Pendulum-v1 -n 250 -optimize --optimization-log-path logs-opt-ddpg --log-folder ddpg --n-trials 5
python train.py --algo td3 --env Pendulum-v1 -n 250 -optimize --optimization-log-path logs-opt-td3 --log-folder td3 --n-trials 5
python train.py --algo sac --env Pendulum-v1 -n 250 -optimize --optimization-log-path logs-opt-sac --log-folder sac --n-trials 5
