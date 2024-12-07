bash run.sh CartPoleSarsa
bash run.sh CartPoleMonteCarlo
bash run.sh CartPoleReinforce
bash run.sh CartPoleReinforceBaseline

python plots.py CartPoleSarsa
python plots.py CartPoleMonteCarlo
python plots.py CartPoleReinforce
python plots.py CartPoleReinforceBaseline
