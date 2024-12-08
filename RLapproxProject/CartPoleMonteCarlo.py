#!/usr/bin/env python
import gymnasium as gym
from torch import nn
import DiscreteMonteCarloAgent as rl


env = gym.make("CartPole-v1", max_episode_steps=100)
dimObs = env.observation_space.shape[0]


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        # BEGIN YOUR CODE HERE
        self.nn = nn.Linear(dimObs, 32)
        self.relu1 = nn.ReLU()
        self.nn2 = nn.Linear(32, 1)
        # END YOUR CODE HERE

    def forward(self, x):
        x = self.nn(x)
        x = self.relu1(x)
        x = self.nn2(x)
        return x


import sys

run = int(sys.argv[1]) if len(sys.argv) == 2 else None

# Play with gamma, alpha, and perhaps other pararameters:
agent = rl.Agent(Q, env.action_space.n, gamma=0.9, alpha=0.0015, epsanneal=10)
agent.train(env)
if run is not None:
    agent.save(f"CartPoleMonteCarlo-{run:02d}.npy")
