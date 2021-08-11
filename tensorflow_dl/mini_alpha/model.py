import tensorflow as tf
import tensorflow.keras.layers as layers

NUM_FILTERS = 64


class Net(tf.keras.Model):
    """
    Alpha zero network include 40 residual blocks and two head:
    Value :     input -> 1 conv filters/1 -> batch normalize -> activation -> fully connected -> activation -> fully connected -> tanh
    Policy:     input -> 2 conv filters/1 -> batch normalize -> activation -> fully connected
    Residual:   input -> 256 conv filters/3 -> batch normalize -> activation -> 256 conv filters/3 -> batch normalize -> skip connection -> activation
    :return return the policy prob for every actions and estimated value-action
    """

    def get_config(self):
        pass

    def __init__(self, actions_n):
        super(Net, self).__init__()

        self.conv_in = tf.keras.Sequential([
            layers.Conv2D(NUM_FILTERS, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(NUM_FILTERS, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ])

        self.res_blocks = [tf.keras.Sequential([
            layers.Conv2D(NUM_FILTERS, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2D(NUM_FILTERS, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
        ]) for _ in range(6)]

        self.value = tf.keras.Sequential([
            layers.Conv2D(1, kernel_size=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(128),
            layers.LeakyReLU(),
            layers.Dense(1),
            layers.Activation("tanh")
        ])

        # Policy net
        self.policy = tf.keras.Sequential([
            layers.Conv2D(2, kernel_size=1),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(actions_n)
        ])

    def call(self, inputs, training=None, mask=None):
        v = self.conv_in(inputs)
        for block in self.res_blocks:
            v = v + block(v)
        val = self.value(v)
        pol = self.policy(v)

        return pol, val
