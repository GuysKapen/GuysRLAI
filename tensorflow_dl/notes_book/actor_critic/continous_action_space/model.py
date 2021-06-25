import tensorflow as tf
import tensorflow.keras.layers as layers

UNITS = 128


class A2C(tf.keras.Model):
    def get_config(self):
        return super(A2C, self).get_config()

    def __init__(self, act_size):
        super(A2C, self).__init__()

        self.base = tf.keras.Sequential([
            layers.Dense(UNITS, activation='relu')
        ])

        self.mu = tf.keras.Sequential([
            layers.Dense(act_size, activation='tanh')
        ])

        self.var = tf.keras.Sequential([
            layers.Dense(act_size, activation='softplus')
        ])

        self.value = layers.Dense(1)

    def call(self, x, training=None, mask=None):
        base_out = self.base(x)

        return self.mu(base_out), self.var(base_out), self.value(base_out)


class DDPGActor(tf.keras.Model):
    def __init__(self, act_size):
        super(DDPGActor, self).__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(act_size, activation='tanh')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.net(inputs)


class DDPGCritic(tf.keras.Model):
    def __init__(self, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
        ])

        self.out_net = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        obs = self.obs_net(x)
        return self.out_net(tf.concat([obs, a], axis=1))


class D4PGCritic(tf.keras.Model):
    def __init__(self, act_size, n_atoms, v_min, v_max):
        super(D4PGCritic, self).__init__()

        self.obs_net = tf.keras.Sequential([
            layers.Dense(512, activation='relu')
        ])

        self.out_net = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(n_atoms)
        ])

        delta = (v_max - v_min) / (n_atoms - 1)

        self.supports = tf.Variable(tf.convert_to_tensor(tf.range(v_min, v_max + delta, delta), dtype=tf.float32), trainable=False)

    def call(self, inputs, training=None, mask=None):
        x, a = inputs
        obs = self.obs_net(x)
        return self.out_net(tf.concat([obs, a], axis=1))

    def distr_to_q(self, distr):
        weights = tf.nn.softmax(distr, axis=1) * self.supports
        res = tf.reduce_sum(weights, axis=-1)
        return tf.expand_dims(res, axis=-1)


