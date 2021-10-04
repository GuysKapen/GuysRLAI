import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import MiniStar_Arch_Hyper_Parameters as MAHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS

from tensorflow_dl.mini_alpha_star.libs import utils

debug = False


class ResBlockFiLM(tf.keras.Model):
    def __init__(self, filter_size):
        super(ResBlockFiLM, self).__init__()

        self.conv1 = layers.Conv2D(filter_size, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.conv2 = layers.Conv2D(filter_size, kernel_size=3, strides=1, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x, gamma, beta = inputs
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        res = self.conv1(x)
        out = self.conv2(res)
        res = tf.transpose(res, perm=[0, 3, 1, 2])
        out = tf.transpose(out, perm=[0, 3, 1, 2])
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

        # The two are concatenated together along the channel dimension
        # map_skip [-1, 128, 16, 16]
        # x [-1, 132, 16, 16]
        print(f"map_slip.shape: {map_skip.shape}") if debug else None
        x = tf.concat([autoregressive_embedding_map, map_skip], axis=1)
        print(f'x.shape: {x.shape}') if debug else None

        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = tf.nn.relu(self.ds_1(tf.nn.relu(x)))
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # 3d tensor (h, w, c) is then passed through series of gated resblocks
        # with 128 channels, kernel size 3 and FiLM, gated on autoregressive_embedding
        # FilM is Feature wise Linear Modulation, paper: FiLM: Visual Reasoning with General Conditioning Layer
        x = self.film_net((x, autoregressive_embedding))

        # x [-1, 128, 16, 16]
        # Using the elements of map_skip i order of last Resblockskip to first
        x = x + map_skip
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        # Afterwards, it is upsampled 2x by each of a series of transpose conv
        # with kernel 4 and channel 128, 64, 16 and 1 respectively
        # Upsampled beyond the 128x128 input to 256x256 target location selection
        x = tf.nn.relu(self.us_1(x))
        x = tf.nn.relu(self.us_2(x))
        x = tf.nn.relu(self.us_3(x))

        if AHP == MAHP:
            x = tf.nn.relu(self.us_4(x))
            # Only in mAS, need more upsample step
            x = tf.nn.relu(self.us_5(x))
        else:
            x = tf.nn.relu(self.us_4_original(x))
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Those final logits are flattened and sampled (masking out invalid locations using action_type
        # Such as those outside the camera for build actions) with temperature .8
        # to get the actual target position
        # x [-1, 1, 256, 256]
        print(f'x.shape: {x.shape}') if debug else None
        y = tf.reshape(x, shape=(batch_size, 1 * self.output_map_size * self.output_map_size))

        # masking out invalid locations using action_type
        mask = tf.ones((batch_size, 1 * self.output_map_size * self.output_map_size))
        print(f'mask: {mask}') if debug else None

        y_2 = y * mask
        print(f"y_2: {y_2}") if debug else None

        target_location_logits = y_2 / self.temperature
        print(f'target_location_logits.shape: {target_location_logits.shape}') if debug else None

        target_location_probs = self.softmax(target_location_logits)
        location_id = tf.random.categorical(target_location_probs, num_samples=1)
        print(f"location_id: {location_id}") if debug else None
        print(f"location_id.shape: {location_id.shape}") if debug else None

        location_out = list(tf.squeeze(location_id, axis=-1).numpy())
        print(f'location_out: {location_out}') if debug else None

        for i, index in enumerate(location_id):
            target_location_y = index // self.output_map_size
            target_location_x = index - self.output_map_size * target_location_y
            print(f'target_location_y: {target_location_y}, target_location_x: {target_location_x}') if debug else None
            location_out[i] = [target_location_y, target_location_x]

        # If action_type does not involve targeting location, this head is ignored
        target_location_mask = utils.action_involve_targeting_location_mask(action_type)
        # target_location_mask: [b x 1]
        print(f'target_location_mask: {target_location_mask}') if debug else None
        location_out = tf.squeeze(tf.convert_to_tensor(location_out))
        print(f"location_out: {location_out}") if debug else None
        print(f'location_out.shape: {location_out.shape}') if debug else None

        target_location_logits = tf.reshape(target_location_logits,
                                            shape=(-1, self.output_map_size, self.output_map_size))
        target_location_logits = target_location_logits * tf.expand_dims(
            tf.cast(target_location_mask, dtype=tf.float32), axis=-1)
        location_out = location_out * tf.cast(target_location_mask, dtype=tf.int64)

        return target_location_logits, location_out


def test():
    batch_size = 2
    autoregressive_embedding = tf.random.normal((batch_size, AHP.autoregressive_embedding_size))
    action_type_sample = 65  # func: 65/Effect_PsiStorm_pt (1/queued [2]; 2/unit_tags [512]; 0/world [0, 0])
    action_type = tf.random.uniform(minval=0, maxval=SFS.available_actions, shape=(batch_size, 1), dtype=tf.int32)

    if AHP == MAHP:
        map_skip = tf.random.normal((batch_size, AHP.location_head_max_map_channels, 8, 8))
    else:
        map_skip = tf.random.normal((batch_size, AHP.location_head_max_map_channels, 16, 16))

    location_head = LocationHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    target_location_logits, target_location = \
        location_head.call((autoregressive_embedding, action_type, map_skip))

    if target_location_logits is not None:
        print("target_location_logits:", target_location_logits) if debug else None
        print("target_location_logits.shape:", target_location_logits.shape) if debug else None
    else:
        print("target_location_logits is None!")

    if target_location is not None:
        print("target_location:", target_location) if debug else None
        # print("target_location.shape:", target_location.shape) if debug else None
    else:
        print("target_location is None!")

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
