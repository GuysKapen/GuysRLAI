import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

from tensorflow_dl.libs.agent import BaseAgent

UNITS = 128


class Actor(tf.keras.Model):
    def get_config(self):
        return super(Actor, self).get_config()

    def __init__(self, act_size):
        super(Actor, self).__init__()

        self.seq = tf.keras.Sequential([
            layers.Dense(UNITS, activation='tanh'),
            layers.Dense(UNITS, activation='tanh'),
            layers.Dense(act_size)
        ])

        self.logstd = tf.Variable(tf.zeros(act_size))

    def call(self, x, training=None, mask=None):
        return self.seq(x)


class Critic(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        return self.value(inputs)

    def __init__(self, obs_size):
        super(Critic, self).__init__()

        self.value = tf.keras.Sequential([
            layers.Dense(UNITS, activation='relu'),
            layers.Dense(UNITS, activation='relu'),
            layers.Dense(1)
        ])


class AgentA2C(BaseAgent):
    def __init__(self, net):
        self.net = net

    def __call__(self, states, agent_states):
        states_v = tf.convert_to_tensor(states, dtype=tf.float32)
        mu_v = self.net(states_v)
        mu = mu_v.numpy()
        logstd = self.net.logstd.numpy()
        actions = mu + tf.math.exp(logstd) * np.random.normal(size=logstd.shape)
        actions = np.clip(actions, -1, 1)
        np.nan_to_num(actions, False)
        return actions, agent_states
