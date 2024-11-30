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
        # BEGIN YOUR CODE HEREobs_space
        state = env.reset()[0]
        action = env.action_space.sample()
        q_value = 0
        G = 0
        T = 0
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)

            next_action, next_q_value = self.chooseAction(
                next_state[0], env.action_space
            )

            q_value = self.q[action](torch.tensor(state[0]))

            G += reward

            # SARSA update
            target = reward + (0 if done else self.gamma * next_q_value)

            self.update(action, target, q_value)

            state = next_state
            action = next_action
            T += 1

        # END YOUR CODE HERE
        return T, G
