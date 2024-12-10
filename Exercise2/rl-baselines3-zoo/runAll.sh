python train.py --algo ddpg --env Pendulum-v1 -n 25000 -optimize --optimization-log-path logs-opt-ddpg --log-folder ddpg --n-trials 50
python train.py --algo td3 --env Pendulum-v1 -n 25000 -optimize --optimization-log-path logs-opt-td3 --log-folder td3 --n-trials 50
python train.py --algo sac --env Pendulum-v1 -n 25000 -optimize --optimization-log-path logs-opt-sac --log-folder sac --n-trials 50
