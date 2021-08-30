import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import MiniStar_Arch_Hyper_Parameters as MAHP

from tensorflow_dl.mini_alpha_star.libs import utils

debug = True


class ResBlockFiLM(tf.keras.Model):
    def __init__(self, filter_size):
        super(ResBlockFiLM, self).__init__()

        self.conv1 = layers.Conv2D(filter_size, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.conv2 = layers.Conv2D(filter_size, kernel_size=3, strides=1, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x, gamma, beta = inputs
        res = self.conv1(x)
        out = self.conv2(res)
        out = self.bn(out)

        gamma = gamma[..., None, None]
        beta = beta[..., None, None]

        out = gamma * out + beta

        out = tf.nn.relu(out)
        out = out + res

        return out


class FiML(tf.keras.Model):
    def __init__(self, n_resblock=4, conv_hidden=128, gate_size=1024):
        super(FiML, self).__init__()
        self.gate_size = gate_size
        self.n_resblock = n_resblock
        self.conv_hidden = conv_hidden

        self.resblocks = [ResBlockFiLM(conv_hidden) for _ in range(n_resblock)]

        self.film_net = layers.Dense(conv_hidden * 2 * n_resblock)

    def call(self, inputs, training=None, mask=None):
        x, gate = inputs
        out = x
        film = tf.split(self.film_net(gate), self.n_resblock * 2, axis=1)

        for i, resblock in enumerate(self.resblocks):
            out = resblock((out, film[i * 2], film[i * 2 + 1]))

        return out


class LocationHead(tf.keras.Model):
    def __init__(self,
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size,
                 output_map_size=SCHP.world_size,
                 is_sl_training=True,
                 max_map_channels=AHP.location_head_max_map_channels,
                 temperature=0.8
                 ):
        super(LocationHead, self).__init__()
        self.is_sl_training = is_sl_training
        if not self.is_sl_training:
            self.temperature = temperature
        else:
            self.temperature = 1.0

        mmc = max_map_channels

        self.ds_1 = layers.Conv2D(mmc, kernel_size=1, strides=1, padding='valid', use_bias=True)

        self.film_net = FiML(n_resblock=4, conv_hidden=mmc, gate_size=autoregressive_embedding_size)

        self.us_1 = layers.Conv2DTranspose(int(mmc / 2), kernel_size=4, strides=2, padding='same', use_bias=True)
        self.us_2 = layers.Conv2DTranspose(int(mmc / 4), kernel_size=4, strides=2, padding='same', use_bias=True)
        self.us_3 = layers.Conv2DTranspose(int(mmc / 8), kernel_size=4, strides=2, padding='same', use_bias=True)
        self.us_4 = layers.Conv2DTranspose(int(mmc / 16), kernel_size=4, strides=2, padding='same', use_bias=True)

        self.us_4_original = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=True)

        self.us_5 = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=True)

        self.output_map_size = output_map_size
        self.softmax = layers.Softmax(axis=-1)

    def preprocess(self):
        pass

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: autoregressive_embedding [b, autoregressive_embedding_size]
        action_type: [b, 1],
        map_skip: [b, entity_size, embedding_size]
        :param training:
        :param mask:
        :return:
        target_location_logits [b, self.output_map_size, self.output_map_size]
        location_out: [b, 1]
        """
        autoregressive_embedding, action_type, map_skip = inputs
        # autoregressive_embedding is reshaped to have the same height / width as the final skip in map_skip
        # which was just before map information was reshaped to a 1D embedding with 4 channels
        batch_size = map_skip.shape[0]

        assert autoregressive_embedding.shape[0] == action_type.shape[0]
        assert autoregressive_embedding.shape[0] == map_skip.shape[0]
        reshape_size = map_skip.shape[-1]
        reshape_channels = int(AHP.autoregressive_embedding_size / (reshape_size * reshape_size))

        print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None
        autoregressive_embedding_map = tf.reshape(autoregressive_embedding,
                                                  shape=(batch_size, -1, reshape_size, reshape_size))
        print("autoregressive_embedding_map.shape:", autoregressive_embedding_map.shape) if debug else None


