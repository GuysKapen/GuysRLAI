import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import nn as F

from tensorflow_dl.mini_alpha_star.libs import utils
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP

debug = False


class TargetUnitHead(tf.keras.Model):
    """
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        target_unit_logits - The logits corresponding to the probabilities of targeting a unit
        target_unit - The sampled target unit
    """

    def __init__(self, embedding_size=AHP.entity_embedding_size,
                 max_number_of_unit_types=SCHP.max_unit_type,
                 is_sl_training=True, temperature=0.8,
                 original_256=AHP.original_256, original_32=AHP.original_32,
                 max_selected=1, autoregressive_embedding_size=AHP.autoregressive_embedding_size):
        super().__init__()
        self.is_sl_training = is_sl_training
        if not self.is_sl_training:
            self.temperature = temperature
        else:
            self.temperature = 1.0

        self.max_number_of_unit_types = max_number_of_unit_types
        self.func_embed = layers.Dense(original_256)  # with relu

        self.conv_1 = layers.Conv1D(original_32, kernel_size=1, strides=1,
                                    padding='valid', use_bias=True)
        self.fc_1 = layers.Dense(original_256)
        self.fc_2 = layers.Dense(original_32)

        self.small_lstm = layers.LSTM(original_32, dropout=0.0, return_state=True, return_sequences=True)

        # We mostly target one unit
        self.max_selected = 1

        self.softmax = layers.Softmax(axis=-1)

    def preprocess(self):
        pass

    def forward(self, autoregressive_embedding, action_type, entity_embeddings):
        """
        Inputs:
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
            action_type: [batch_size x 1]
            entity_embeddings: [batch_size x entity_size x embedding_size]
        Output:
            target_unit_logits: [batch_size x max_selected x entity_size]
            target_unit: [batch_size x max_selected x 1]
        """

        batch_size = entity_embeddings.shape[0]
        assert autoregressive_embedding.shape[0] == action_type.shape[0]
        assert autoregressive_embedding.shape[0] == entity_embeddings.shape[0]
        # entity_embeddings shape is [batch_size x entity_size x embedding_size]
        entity_size = entity_embeddings.shape[-2]

        # `func_embed` is computed the same as in the Selected Units head,
        # and used in the same way for the query (added to the output of the `autoregressive_embedding`
        # passed through a linear of size 256).
        unit_types_one_hot = utils.action_can_apply_to_entity_types_mask(action_type)

        assert unit_types_one_hot.shape[-1] == self.max_number_of_unit_types
        # unit_types_mask shape: [batch_size x self.max_number_of_unit_types]
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))
        # the_func_embed shape: [batch_size x 256]
        print("the_func_embed:", the_func_embed) if debug else None
        print("the_func_embed.shape:", the_func_embed.shape) if debug else None

        # Because we mostly target one unit, we don't need a mask.

        # The query is then passed through a ReLU and a linear of size 32,
        # and the query is applied to the keys which are created the
        # same way as in the Selected Units head to get `target_unit_logits`.
        # input : [batch_size x entity_size x embedding_size]
        key = self.conv_1(entity_embeddings)
        # output : [batch_size x entity_size x key_size], note key_size = 32
        print("key:", key) if debug else None
        print("key.shape:", key.shape) if debug else None

        target_unit_logits_list = []
        target_unit_list = []
        hidden = None

        # note: repeated for selecting up to one unit
        max_selected = self.max_selected
        for i in range(max_selected):
            x = self.fc_1(autoregressive_embedding)
            print("x:", x) if debug else None
            print("x.shape:", x.shape) if debug else None

            assert the_func_embed.shape == x.shape
            z_1 = the_func_embed + x
            print("z_1:", z_1) if debug else None
            print("z_1.shape:", z_1.shape) if debug else None

            z_2 = self.fc_2(z_1)
            print("z_2:", z_2) if debug else None
            print("z_2.shape:", z_2.shape) if debug else None
            z_2 = tf.expand_dims(z_2, axis=1)

            # The result is fed into a LSTM with size 32 and zero initial state to get a query.
            if i == 0:
                query, *hidden = self.small_lstm(z_2)
            else:
                query, *hidden = self.small_lstm(z_2, hidden)
            print("query:", query) if debug else None
            print("query.shape:", query.shape) if debug else None

            # AlphaStar: The query is then passed through a ReLU and a linear of size 32,
            # AlphaStar: and the query is applied to the keys which are created the same way as in
            # AlphaStar: the Selected Units head to get `target_unit_logits`.

            # below is matrix multiply
            # key_shape: [batch_size x entity_size x key_size], note key_size = 32
            # query_shape: [batch_size x seq_len x hidden_size], note hidden_size is also 32, seq_len = 1
            y = tf.matmul(key, query, transpose_b=True)
            print("y:", y) if debug else None
            print("y.shape:", y.shape) if debug else None
            y = tf.squeeze(y, axis=-1)
            # y shape: [batch_size x entity_size]

            target_unit_logits = y / self.temperature
            # target_unit_logits shape: [batch_size x entity_size]
            print("target_unit_logits:", target_unit_logits) if debug else None
            print("target_unit_logits.shape:", target_unit_logits.shape) if debug else None

            target_unit_probs = self.softmax(target_unit_logits)
            # target_unit_probs shape: [batch_size x entity_size]

            # AlphaStar: `target_unit` is sampled from `target_unit_logits` using a multinomial with temperature 0.8.
            target_unit_id = tf.random.categorical(target_unit_probs, 1)
            # target_unit_id shape: [batch_size x 1]
            print("target_unit_id:", target_unit_id) if debug else None
            print("target_unit_id.shape:", target_unit_id.shape) if debug else None

            # note, we add a dimension where is in the seq_one to help
            # we concat to the one : [batch_size x max_selected x ?]
            target_unit_logits_list.append(tf.expand_dims(target_unit_logits, axis=-2))
            target_unit_list.append(tf.expand_dims(target_unit_id, axis=-2))

            # Note that since this is one of the two terminal arguments (along
            # with Location Head, since no action has both a target unit and a
            # target location), it does not return `autoregressive_embedding`.

            # note: we only select one unit, so return the first one
            target_unit_logits = tf.concat(target_unit_logits_list, axis=1)
            # target_unit_logits: [batch_size x max_selected x entity_size]
            target_unit = tf.concat(target_unit_list, axis=1)
            # target_units: [batch_size x max_selected x 1]

            # AlphaStar: If `action_type` does not involve targeting units, this head is ignored.
            target_unit_mask = utils.action_involve_targeting_units_mask(action_type)
            # target_unit_mask: [batch_size x 1]
            print("target_unit_mask:", target_unit_mask) if debug else None

            print("target_unit_logits.shape:", target_unit_logits.shape) if debug else None
            print("target_unit_mask.shape:", target_unit_mask.shape) if debug else None
            target_unit_logits = target_unit_logits * tf.expand_dims(tf.cast(target_unit_mask, dtype=tf.float32),
                                                                     axis=-1)
            print("target_unit.shape:", target_unit.shape) if debug else None
            target_unit = target_unit * tf.expand_dims(tf.cast(target_unit_mask, dtype=tf.int64), axis=-1)

            return target_unit_logits, target_unit

    def call(self, inputs, training=None, mask=None):
        autoregressive_embedding, action_type, entity_embeddings = inputs
        return self.forward(autoregressive_embedding, action_type=action_type, entity_embeddings=entity_embeddings)
    

def test():
    action_type_sample = 352  # func: 352/Effect_WidowMineAttack_unit (1/queued [2]; 2/unit_tags [512]; 3/target_unit_tag [512])

    batch_size = 2
    autoregressive_embedding = tf.random.normal((batch_size, AHP.autoregressive_embedding_size))
    action_type = tf.random.uniform(minval=0, maxval=SFS.available_actions, shape=(batch_size, 1), dtype=tf.int32)
    entity_embeddings = tf.random.normal((batch_size, AHP.max_entities, AHP.entity_embedding_size))

    target_units_head = TargetUnitHead()

    print("autoregressive_embedding:", autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None

    target_unit_logits, target_unit = \
        target_units_head.forward(autoregressive_embedding, action_type, entity_embeddings)

    if target_unit_logits is not None:
        print("target_unit_logits:", target_unit_logits) if debug else None
        print("target_unit_logits.shape:", target_unit_logits.shape) if debug else None
    else:
        print("target_unit_logits is None!")

    if target_unit is not None:
        print("target_unit:", target_unit) if debug else None
        print("target_unit.shape:", target_unit.shape) if debug else None
    else:
        print("target_unit is None!")

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
