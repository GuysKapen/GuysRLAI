import tensorflow as tf
import numpy as np
from tensorflow_dl.libs.agent import BaseAgent


class AgentA2C(BaseAgent):
    def __init__(self, net):
        self.net = net

    def __call__(self, states, agent_states):
        states_v = tf.convert_to_tensor(states, dtype=tf.float32)
        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.numpy()
        sigma = tf.math.sqrt(var_v)
        actions = np.random.normal(mu, sigma)
        actions = np.clip(actions, -1, 1)
        np.nan_to_num(actions, False)
        return actions, agent_states


class AgentDDPG(BaseAgent):
    def __init__(self, net, ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_sigma = ou_sigma
        self.ou_teta = ou_teta
        self.ou_epsilon = ou_epsilon

    def __call__(self, states, agent_states):
        states_v = tf.convert_to_tensor(states, dtype=tf.float32)
        mu_v = self.net(states_v)
        actions = mu_v.numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_action_states = []
            for action_state, action in zip(agent_states, actions):
                if action_state is None:
                    action_state = np.zeros(shape=action.shape, dtype=np.float32)
                action_state += self.ou_teta * (self.ou_mu - action_state)
                action_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * action_state
                new_action_states.append(action_state)
        else:
            new_action_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_action_states
