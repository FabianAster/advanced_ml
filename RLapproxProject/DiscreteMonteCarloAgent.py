# Base class for RL agents for episodic tasks
# - derived from DiscreteAgent

import torch
import DiscreteAgent as Discrete


class Agent(Discrete.Agent):
    def __init__(self, Q, nActions, gamma=1, **kwargs):
        super(Agent, self).__init__(Q, nActions, **kwargs)
        self.gamma = gamma

    # This method trains the agent for one episode on the given
    # gymnasium environment, and is called by the base class' train() method.
    # This method should repeatedly call chooseAction() and update().
    # This method must return
    #   T, the length of this episode in time steps, and
    #   G, the (discounted) return earned during this episode.
    def trainEpisode(self, env):
        # BEGIN YOUR CODE HERE
        state, _ = env.reset()
        episode = []
        done = False
        T = 0

        while not done and T < 100:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            T += 1

        G = 0
        for state, action, reward in reversed(episode):
            G = reward + G  # Accumulate return
            q_value = self.q[action](torch.tensor(state))
            target = G
            self.update(action, target, q_value)

        # END YOUR CODE HERE
        return T, G
