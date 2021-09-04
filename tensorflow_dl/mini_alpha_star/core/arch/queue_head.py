import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.libs import utils

from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS

debug = True


class QueueHead(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, input_size=AHP.autoregressive_embedding_size, original_256=AHP.original_256,
                 max_queue=SFS.last_repeat_queued, is_sl_training=True,
                 temperature=.8):
        """
        :param input_size:
        :param original_256:
        :param max_queue:
        :param is_sl_training:
        :param temperature:
        :returns queued_logits - the logits corresponding to the probabilities of queueing and not queueing
        queued - whether or no to queue this action
        autoregressive_embedding - embedding that combines information from lstm_output and all previous sampled arguments
        """
        super(QueueHead, self).__init__()
        self.is_sl_training = is_sl_training
        if not self.is_sl_training:
            self.temperature = temperature
        else:
            self.temperature = 1.0

        self.fc_1 = layers.Dense(original_256)  # with relu
        self.fc_2 = layers.Dense(original_256)  # with relu
        self.max_queue = max_queue

        self.embed_fc = layers.Dense(max_queue)  # no relu

        self.fc_3 = layers.Dense(original_256)  # with relu
        self.fc_4 = layers.Dense(original_256)  # with relu
        self.project = layers.Dense(AHP.autoregressive_embedding_size)

        self.relu = layers.ReLU()
        self.softmax = layers.Softmax(axis=-1)

    def preprocess(self):
        pass

    # QUESTION: It is similar to delay head. But how did it use the embedded_entity?
    def forward(self, autoregressive_embedding, action_type, embedded_entity=None):
        # AlphaStar: Queued Head is similar to the delay head except a temperature of 0.8
        # AlphaStar: is applied to the logits before sampling,
        x = self.fc_1(autoregressive_embedding)
        print("x.shape:", x.shape) if debug else None
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        # note: temperature is used here, compared to delay head
        queue_logits = self.embed_fc(x) / self.temperature
        queue_probs = self.softmax(queue_logits)
        # AlphaStar: the size of `queued_logits` is 2 (for queueing and not queueing),
        queue = tf.random.categorical(queue_probs, 1)

        # similar to action_type here, change it to one_hot version
        queue_one_hot = utils.one_hot_embedding(queue, self.max_queue)
        # to make the dim of queue_one_hot as queue
        queue_one_hot = tf.squeeze(queue_one_hot, axis=-2)

        z = self.relu(self.fc_3(queue_one_hot))
        z = self.relu(self.fc_4(z))
        t = self.project(z)
        # make sure autoregressive_embedding has the same shape as y, prevent the auto broadcasting
        assert autoregressive_embedding.shape == t.shape

        # AlphaStar: and the projected `queued` is not added to `autoregressive_embedding`
        # AlphaStar: if queuing is not possible for the chosen `action_type`
        # note: projected `queued` is not added to `autoregressive_embedding` if queuing is not
        # possible for the chosen `action_type`

        assert action_type.shape[0] == autoregressive_embedding.shape[0]
        mask = tf.cast(utils.action_can_be_queued_mask(action_type), dtype=tf.float32)
        print("mask:", mask) if debug else None
        autoregressive_embedding = autoregressive_embedding + mask * t

        ''' # below code only consider the cases when action_type is scalar
        if L.action_can_be_queued(action_type):
            autoregressive_embedding = autoregressive_embedding + t
        else:
            print("None add to autoregressive_embedding!") if debug else None
        '''

        return queue_logits, queue, autoregressive_embedding

    def call(self, inputs, training=None, mask=None, **kwargs):
        return self.forward(*inputs, **kwargs)


def test():
    batch_size = 2
    autoregressive_embedding = tf.random.normal((batch_size, AHP.autoregressive_embedding_size))
    action_type = tf.random.uniform(minval=0, maxval=SFS.available_actions, shape=(batch_size, 1), dtype=tf.int32)
    queue_head = QueueHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    queue_logits, queue, autoregressive_embedding = queue_head((autoregressive_embedding, action_type))

    print("queue_logits:", queue_logits) if debug else None
    print("queue_logits.shape:", queue_logits.shape) if debug else None
    print("queue:", queue) if debug else None
    print("queue.shape:", queue.shape) if debug else None
    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()