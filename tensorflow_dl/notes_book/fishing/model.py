import tensorflow as tf
import tensorflow.keras.layers as layers


class FishingA2C(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, n_actions):
        super(FishingA2C, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            layers.Flatten()
        ])

        self.policy = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(n_actions)
        ])

        self.value = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, x, training=None, mask=None):
        fx = tf.cast(x, tf.float32) / 256.0
        conv_out = self.conv(fx)
        return self.policy(conv_out), self.value(conv_out)
