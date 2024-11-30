#!/usr/bin/env python
import gymnasium as gym
from torch import nn
import DiscreteSarsaAgent as rl


env = gym.make("CartPole-v1", max_episode_steps=100)
dimObs = env.observation_space.shape[0]


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        # BEGIN YOUR CODE HERE
        self.nn = nn.Linear(1, 4)
        self.nn1 = nn.Linear(4, 10)
        self.nn2 = nn.Linear(10, 1)
        self.softmax = nn.Softmax(0)
        # END YOUR CODE HERE

    def forward(self, x):
        x = self.nn(x.flatten())
        x = self.nn1(x)
        x = self.softmax(x)
        x = self.nn2(x)
        return x


import sys

run = int(sys.argv[1]) if len(sys.argv) == 2 else None

# Play with gamma, alpha, and perhaps other pararameters:
agent = rl.Agent(Q, env.action_space.n, gamma=1, alpha=0.0001)
agent.train(env)
if run is not None:
    agent.save(f"CartPoleSarsa-{run:02d}.npy")
