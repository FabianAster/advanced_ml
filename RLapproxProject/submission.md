# My Observations and Comments

Somehow my MonteCarlo refuses to learn, but i cant find any errors. By executing and testing it a lot, ive discovered that very rarely it does learn but forgets the learned things again.
Also for SARSA i struggled to keep the things learned. Using the annealEps and a lower learning rate its solvable, but for monteCarlo i couldnt find a working solution. Still in sarsa its minimally visible at the end of the graph.

Reinforce and Reinforce with baseline look ok, but i ran out of tuning time for them, since each meaningful run.sh takes about a hour per method.

For the neural network i found that a network with two fully connected layers and a RELU activation function worked best. It learns fast but takes a while to train.
