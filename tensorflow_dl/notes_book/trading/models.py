import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np


class NoisyLinear(tf.keras.layers.Dense):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(out_features, use_bias=bias)
        self.sigma_weight = tf.Variable(shape=(out_features, in_features), initial_value=sigma_init)
        self.in_features = in_features
        self.epsilon_weight = tf.Variable(shape=(out_features, in_features), trainable=False)
        if bias:
            self.sigma_bias = tf.Variable(shape=(out_features,), initial_value=sigma_init)
            self.epsilon_bias = tf.Variable(shape=(out_features, in_features), trainable=False)

    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        self.set_weights(
            [tf.random.uniform(shape=self.weights[0].shape, minval=-std, maxval=std),
             tf.random.uniform(shape=self.weights[1].shape, minval=-std, maxval=std)])

    def call(self, inputs):
        self.sigma_weight = tf.random.normal(shape=self.sigma_weight.shape)
        bias = self.weights[1]
        if bias is not None:
            self.epsilon_bias = tf.random.normal(shape=self.epsilon_bias.shape)
            bias = bias + self.sigma_weight * self.epsilon_bias
        return tf.matmul(input, self.weights[0] + self.sigma_weight * self.epsilon_weight, transpose_b=True) + bias


class SimpleFFDQN(tf.keras.Model):
    def __init__(self):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1)
        ])

        self.fc_adv = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1)
        ])

    def call(self, x, training=None, mask=None):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)

    def get_config(self):
        return super(SimpleFFDQN, self).get_config()


class DQNConv1D(tf.keras.Model):
    def __init__(self):
        super(DQNConv1D, self).__init__()

        self.conv = tf.keras.Sequential([
            layers.Conv1D(128, 5, activation='relu'),
            layers.Conv1D(128, 5, activation='relu'),
        ])

        self.fc_val = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1)
        ])

        self.fc_adv = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1)
        ])

    def call(self, x, training=None, mask=None):
        conv_out = tf.reshape(self.conv(x), shape=(-1,))
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)

    def get_config(self):
        return super(DQNConv1D, self).get_config()


class DQNConv1DLarge(tf.keras.Model):
    def __init__(self):
        super(DQNConv1DLarge, self).__init__()

        self.conv = tf.keras.Sequential([
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPool1D(3, 2),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPool1D(3, 2),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPool1D(3, 2),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPool1D(3, 2),
            layers.Conv1D(32, 3, activation='relu'),
            layers.Conv1D(32, 3, activation='relu'),
        ])

        self.fc_val = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1)
        ])

        self.fc_adv = tf.keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(1)
        ])

    def call(self, x, training=None, mask=None):
        conv_out = tf.reshape(self.conv(x), shape=(-1,))
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - tf.reduce_mean(adv, axis=1, keepdims=True)

    def get_config(self):
        return super(DQNConv1DLarge, self).get_config()
