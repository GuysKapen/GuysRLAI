from abc import ABC, abstractmethod

import tensorflow as tf


def mlp(input_shape, layer_shapes, output_shape, output_activation=tf.identity, activation=tf.nn.elu):
    sizes = [input_shape] + layer_shapes + [output_shape]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [tf.keras.layers.Dense(input_shape=sizes[i], units=sizes[i + 1]), act()]
    return tf.keras.Sequential([*layers])


class AbstractNet(ABC, tf.keras.Model):
    def __init__(self):
        super(AbstractNet, self).__init__()
        pass

    @abstractmethod
    def init_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass


class MuZeroFullyConnectedNet(AbstractNet):
    def __init__(self,
                 obs_shape,
                 stacked_observations,
                 action_space_shape,
                 encoding_shape,
                 fc_reward_layers,
                 fc_value_layers,
                 fc_policy_layers,
                 fc_representation_layers,
                 fc_dynamics_layers,
                 supper_shape):
        super(MuZeroFullyConnectedNet, self).__init__()
        self.action_space_size = action_space_shape
        self.full_support_size = 2 * supper_shape + 1

        self.representation_network = mlp(
            obs_shape[0]
            * obs_shape[1]
            * obs_shape[2]
            * (stacked_observations + 1)
            + stacked_observations
            * obs_shape[1] * obs_shape[2],
            fc_representation_layers,
            encoding_shape
        )

        self.dynamics_encoded_state_network = mlp(
            encoding_shape + self.action_space_size,
            fc_dynamics_layers,
            encoding_shape
        )

        self.dynamics_reward_network = mlp(
            encoding_shape,
            fc_reward_layers,
            self.full_support_size
        )

        self.prediction_policy_network = mlp(
            encoding_shape,
            fc_policy_layers,
            self.action_space_size
        )

        self.prediction_value_network = mlp(
            encoding_shape,
            fc_value_layers,
            self.full_support_size
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(tf.reshape(observation, shape=(observation.shape[0], -1)))
        min_encoded_state = tf.reduce_min(encoded_state, axis=1, keepdims=True)
        max_encoded_state = tf.reduce_max(encoded_state, axis=1, keepdims=True)
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (encoded_state - min_encoded_state) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix)
        action_one_hot = (tf.zeros(shape=(action.shape[0], self.action_space_size), dtype=tf.float32))
        tf.tensor_scatter_nd_update(action_one_hot, [1], 1.0)
        x = tf.concat([encoded_state, action_one_hot], axis=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)
        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between 0, 1
        min_next_encoded_state = tf.reduce_min(next_encoded_state, axis=1)
        max_next_encoded_state = tf.reduce_max(next_encoded_state, axis=1)
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (next_encoded_state - min_next_encoded_state) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def init_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)

        # reward  equal to 0 for consistency
        reward = tf.math.log(
            tf.repeat(
                tf.tensor_scatter_nd_update(tf.zeros(shape=(1, self.full_support_size)),
                                            tf.convert_to_tensor([[self.full_support_size // 2]], dtype=tf.int64), 1.0),
                len(observation), axis=1
            )
        )

        return value, reward, policy_logits, encoded_state

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


def conv3x3(channels, stride=1):
    return tf.keras.layers.Conv2D(channels, kernel_size=3, strides=stride, padding='same', use_bias=False)


class ResidualBlock(tf.keras.Model):
    def __init__(self, num_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.num_channels = num_channels
        self.stride = stride
        self.conv1 = conv3x3(channels=num_channels, stride=stride)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = conv3x3(num_channels)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += inputs
        out = tf.nn.relu(out)
        return out

    def get_config(self):
        return {'num_channels': self.num_channels, 'stride': self.stride}


class DownSample(tf.keras.Model):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels // 2,
                                            kernel_size=3,
                                            strides=2,
                                            padding='same',
                                            use_bias=False)
        self.res_blocks1 = [ResidualBlock(channels // 2) for _ in range(2)]
        self.conv2 = tf.keras.layers.Conv2D(
            channels,
            kernel_size=3,
            strides=2,
            padding='same',
            use_bias=False
        )

        self.res_blocks2 = [ResidualBlock(channels) for _ in range(3)]

        self.pooling1 = tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding=1)
        self.res_blocks3 = [ResidualBlock(channels) for _ in range(3)]

        self.pooling2 = tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding=1)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        for block in self.res_blocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.res_blocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.res_blocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(tf.keras.Model):
    def __init__(self, in_channels, out_channels, h_w):
        super(DownsampleCNN, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(mid_channels, kernel_size=h_w[0] * 2, strides=4,
                                   padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Conv2D(out_channels, kernel_size=5, padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2)
        ])

        self.avg_pool = tf.keras.layers.AvgPool2D(h_w)

    def call(self, inputs, training=None, mask=None):
        x = self.features(inputs)
        x = self.avg_pool(x)
        return x

    def get_config(self):
        return {}
