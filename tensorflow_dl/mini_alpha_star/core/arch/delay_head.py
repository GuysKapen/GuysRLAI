import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.libs import utils
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS

debug = True


def check_nan_and_inf(val, name):
    if tf.reduce_any(tf.math.is_nan(tf.cast(val, dtype=tf.float32))).numpy():
        print(name, 'Find nan:', val)
    if tf.reduce_any(tf.math.is_inf(tf.cast(val, dtype=tf.float32))).numpy():
        print(name, 'Find inf:', val)


class DelayHead(tf.keras.Model):

    def __init__(self, auto_regressive_embedding_size=AHP.autoregressive_embedding_size,
                 original_256=AHP.original_256, max_delay=SFS.last_delay):
        """

        :param auto_regressive_embedding_size:
        :param original_256:
        :param max_delay:
        outputs: delay_logits - the logits corresponding to the probs of each delay
        delay - the sampled delay
        auto_regressive_embedding - embedding that combines info from lstm_output and all previous sampled arguments
        """
        super(DelayHead, self).__init__()
        self.fc_1 = layers.Dense(original_256, activation='relu')
        self.fc_2 = layers.Dense(original_256, activation='relu')
        self.max_delay = max_delay

        self.embed_fc = layers.Dense(max_delay)

        self.fc_3 = layers.Dense(original_256, activation='relu')
        self.fc_4 = layers.Dense(original_256, activation='relu')
        self.project = layers.Dense(auto_regressive_embedding_size)

        self.softmax = layers.Softmax(axis=-1)

    def call(self, auto_regressive_embedding, training=None, mask=None):
        check_nan_and_inf(auto_regressive_embedding, 'auto_regressive_embedding')
        # AlphaStar: `autoregressive_embedding` is decoded using a 2-layer (each with size 256)
        # AlphaStar: linear network with ReLUs,

        x = self.fc_1(auto_regressive_embedding)
        print(f'x.shape {x.shape}') if debug else None
        check_nan_and_inf(x, 'x')
        x = self.fc_2(x)
        check_nan_and_inf(x, 'x')

        # before being embedded into delay_logits that has size 128
        # (one for each delay)
        # possible requested delay in game steps
        # no temperature used
        delay_logits = self.embed_fc(x)
        check_nan_and_inf(delay_logits, 'delay_logits')

        # delay is sampled from delay_logits using a multinomial, though unlink all other arguments
        # no temperature used before sampling

        delay_probs = self.softmax(delay_logits)
        delay = tf.random.categorical(delay_probs, 1)
        check_nan_and_inf(delay, 'delay')
        print(f'delay: {delay}') if debug else None

        # Similar to action_type, delay is projected to a 1D tensor of size 1024 through
        # 2-layer linear network with relu and added to auto_regressive_embedding
        # similar to action_type here, change it to one_hot version
        delay_one_hot = utils.one_hot_embedding(delay, self.max_delay)

        # make the axis of delay_one_hot as delay
        print(f'delay_one_hot: {delay_one_hot}') if debug else None
        print(f'delay_one_hot.shape: {delay_one_hot.shape}') if debug else None
        z = self.fc_3(delay_one_hot)
        z = self.fc_4(z)
        t = self.project(z)

        y = auto_regressive_embedding + t
        assert auto_regressive_embedding.shape == y.shape
        auto_regressive_embedding = y

        return delay_logits, delay, auto_regressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = tf.random.normal([batch_size, AHP.autoregressive_embedding_size])
    delay_head = DelayHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    delay_logits, delay, autoregressive_embedding = delay_head(autoregressive_embedding)

    print("delay_logits:", delay_logits) if debug else None
    print("delay_logits.shape:", delay_logits.shape) if debug else None
    print("delay:", delay) if debug else None
    print("delay.shape:", delay.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
