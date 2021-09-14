import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import nn as F

from tensorflow_dl.mini_alpha_star.libs import utils
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP

debug = True


class SelectedUnitsHead(tf.keras.Model):
    """
    Inputs: autoregressive_embedding, action_type, entity_embeddings
    Outputs:
        units_logits - The logits corresponding to the probabilities of selecting each unit,
            repeated for each of the possible 64 unit selections
        units - The units selected for this action.
        autoregressive_embedding - Embedding that combines information from `lstm_output` and all previous sampled arguments.
    """

    def __init__(self, embedding_size=AHP.entity_embedding_size,
                 max_number_of_unit_types=SCHP.max_unit_type, is_sl_training=True,
                 temperature=0.8, max_selected=AHP.max_selected,
                 original_256=AHP.original_256, original_32=AHP.original_32,
                 autoregressive_embedding_size=AHP.autoregressive_embedding_size):
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

        self.max_selected = max_selected

        self.project = layers.Dense(autoregressive_embedding_size)
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
            units_logits: [batch_size x max_selected x entity_size]
            units: [batch_size x max_selected x 1]
            autoregressive_embedding: [batch_size x autoregressive_embedding_size]
        """

        batch_size = entity_embeddings.shape[0]
        assert autoregressive_embedding.shape[0] == action_type.shape[0]
        assert autoregressive_embedding.shape[0] == entity_embeddings.shape[0]

        # entity_embeddings shape is [batch_size x entity_size x embedding_size]
        entity_size = entity_embeddings.shape[-2]

        # If applicable, Selected Units Head first determines which entity types can accept `action_type`,
        # creates a one-hot of that type with maximum equal to the number of unit types,
        # and passes it through a linear of size 256 and a ReLU. This will be referred to in this head as `func_embed`.
        # QUESTION: one unit type or serveral unit types?
        # ANSWER: serveral unit types, each for one-hot
        # This is some places which introduce much human knowledge
        unit_types_one_hot = utils.action_can_apply_to_entity_types_mask(action_type)
        print("unit_types_one_hot:", unit_types_one_hot) if debug else None

        assert unit_types_one_hot.shape[-1] == self.max_number_of_unit_types
        # unit_types_mask shape: [batch_size x self.max_number_of_unit_types]
        the_func_embed = F.relu(self.func_embed(unit_types_one_hot))

        # the_func_embed shape: [batch_size x 256]
        print("the_func_embed:", the_func_embed) if debug else None
        print("the_func_embed.shape:", the_func_embed.shape) if debug else None

        # It also computes a mask of which units can be selected,
        # initialised to allow selecting all entities that exist(including enemy units).
        mask = tf.ones((batch_size, entity_size))
        print("mask:", mask) if debug else None
        print("mask.shape:", mask.shape) if debug else None

        # It then computes a key corresponding to each entity by feeding `entity_embeddings`
        # through a 1D convolution with 32 channels and kernel size 1,
        # and creates a new variable corresponding to ending unit selection.
        print("entity_embeddings.shape:", entity_embeddings.shape) if debug else None
        # input : [batch_size x entity_size x embedding_size]
        key = self.conv_1(entity_embeddings)
        # output : [batch_size x entity_size x key_size], note key_size = 32
        print("key:", key) if debug else None
        print("key.shape:", key.shape) if debug else None

        # TODO: creates a new variable corresponding to ending unit selection.
        # QUESTION: how to do that?
        units_logits = []
        units = []
        hidden = None

        # AlphaStar: repeated for selecting up to 64 units
        max_selected = self.max_selected
        for i in range(max_selected):
            # AlphaStar: the network passes `autoregressive_embedding` through a linear of size 256,
            # autoregressive_embedding shape: [batch_size x autoregressive_embedding_size]
            x = self.fc_1(autoregressive_embedding)
            # x shape: [batch_size x 256]
            print("x:", x) if debug else None
            print("x.shape:", x.shape) if debug else None

            # AlphaStar: adds `func_embed`,
            assert the_func_embed.shape == x.shape
            z_1 = the_func_embed + x
            # z_1 shape: [batch_size x 256]
            print("z_1:", z_1) if debug else None
            print("z_1.shape:", z_1.shape) if debug else None

            # AlphaStar: and passes the combination through a ReLU and a linear of size 32.
            # note: original writing is wrong, z_2 = F.relu(self.fc_2(z_1))
            # change to below line:
            z_2 = self.fc_2(F.relu(z_1))

            # z_2 shape: [batch_size x 32]
            print("z_2:", z_2) if debug else None
            print("z_2.shape:", z_2.shape) if debug else None

            z_2 = tf.expand_dims(z_2, axis=1)
            # z_2 shape: [batch_size x seq_len x 32], note seq_len = 1

            # AlphaStar: The result is fed into a LSTM with size 32 and zero initial state to get a query.
            if i == 0:
                query, *hidden = self.small_lstm(z_2)
            else:
                query, *hidden = self.small_lstm(z_2, initial_state=hidden)

            print("query:", query) if debug else None
            print("query.shape:", query.shape) if debug else None

            # AlphaStar: The entity keys are multiplied by the query, and
            # AlphaStar: are sampled using the mask and temperature 0.8 to decide which entity to select.

            # note: below is dot product, but I found the shape don't match
            # assert query.shape == key.shape
            # y = key * query

            # below is batch matrix multiply
            # key_shape: [batch_size x entity_size x key_size], note key_size = 32
            # query_shape: [batch_size x seq_len x hidden_size], note hidden_size is also 32, seq_len = 1
            y = tf.matmul(key, query, transpose_b=True)
            # y shape: [batch_size x entity_size x seq_len], note seq_len = 1
            print("y:", y) if debug else None
            print("y.shape:", y.shape) if debug else None
            y = tf.squeeze(y, axis=-1)
            # y shape: [batch_size x entity_size]
            # mask shape: [batch_size x entity_size]
            assert y.shape == mask.shape

            y_2 = y * tf.stop_gradient(mask)
            # y_2 shape: [batch_size x entity_size]
            print("y_2:", y_2) if debug else None
            print("y_2.shape:", y_2.shape) if debug else None

            entity_logits = y_2 / self.temperature
            # entity_logits shape: [batch_size x entity_size]
            print("entity_logits:", entity_logits) if debug else None
            print("entity_logits.shape:", entity_logits.shape) if debug else None

            entity_probs = self.softmax(entity_logits)
            # entity_probs shape: [batch_size x entity_size]

            entity_id = tf.random.categorical(entity_probs, 1)
            # TODO: Wenhai: If this entity_id is a EOF, end the selection

            # entity_id shape: [batch_size x 1]
            print("entity_id:", entity_id) if debug else None
            print("entity_id.shape:", entity_id.shape) if debug else None

            # note, we add a dimension where is in the seq_one to help
            # we concat to the one : [batch_size x max_selected x ?]
            units_logits.append(tf.expand_dims(entity_logits, axis=-2))
            units.append(tf.expand_dims(entity_id, axis=-2))

            # AlphaStar: That entity is masked out so that it cannot be selected in future iterations.
            new_mask = []
            for x, y in zip(entity_id, mask):
                new_mask.append(tf.tensor_scatter_nd_update(y, [x], [0]))
            mask = tf.convert_to_tensor(new_mask)
            print("mask:", mask) if debug else None
            print("mask.shape:", mask.shape) if debug else None

            # AlphaStar: The one-hot position of the selected entity is multiplied by the keys,
            entity_one_hot = tf.squeeze(utils.to_one_hot(entity_id, entity_size), axis=-2)
            # entity_one_hot shape: [batch_size x entity_size]
            print("entity_one_hot:", entity_one_hot) if debug else None
            print("entity_one_hot.shape:", entity_one_hot.shape) if debug else None

            entity_one_hot_unsqueeze = tf.expand_dims(entity_one_hot, axis=-2)
            # entity_one_hot_unsqueeze shape: [batch_size x seq_len x entity_size], note seq_len =1
            # key_shape: [batch_size x entity_size x key_size], note key_size = 32
            out = tf.squeeze(tf.matmul(entity_one_hot_unsqueeze, key), axis=-2)
            # out shape: [batch_size x key_size]
            print("out:", out) if debug else None
            print("out.shape:", out.shape) if debug else None

            # AlphaStar: reduced by the mean across the entities,
            # should be key mean
            # TODO: change to the key mean
            mean = tf.reduce_mean(entity_one_hot, axis=-1, keepdims=True)
            # mean shape: [batch_size x 1]
            out = out - mean
            print("out:", out) if debug else None
            print("out.shape:", out.shape) if debug else None

            # AlphaStar: passed through a linear layer of size 1024,
            t = self.project(out)
            print("t:", t) if debug else None
            print("t.shape:", t.shape) if debug else None

            # AlphaStar: and added to `autoregressive_embedding` for subsequent iterations.
            assert autoregressive_embedding.shape == t.shape
            autoregressive_embedding = autoregressive_embedding + t
            print("autoregressive_embedding:",
                  autoregressive_embedding) if debug else None
            print("autoregressive_embedding.shape:",
                  autoregressive_embedding.shape) if debug else None

        units_logits = tf.concat(units_logits, axis=1)
        # units_logits: [batch_size x max_selected x entity_size]
        units = tf.concat(units, axis=1)
        # units: [batch_size x max_selected x 1]

        # autoregressive_embedding: [batch_size x autoregressive_embedding_size]

        # AlphaStar: If `action_type` does not involve selecting units, this head is ignored.
        # TODO: fix the ignored
        select_unit_mask = utils.action_involve_selecting_units_mask(action_type)
        # select_unit_mask: [batch_size x 1]
        print("select_unit_mask:", select_unit_mask) if debug else None

        print("units_logits.shape:", units_logits.shape) if debug else None
        print("select_unit_mask.shape:", select_unit_mask.shape) if debug else None
        units_logits = units_logits * tf.expand_dims(tf.cast(select_unit_mask, dtype=tf.float32), axis=-1)
        print("units.shape:", units.shape) if debug else None
        units = units * tf.expand_dims(tf.cast(select_unit_mask, dtype=tf.int64), axis=-1)
        print("autoregressive_embedding.shape:", autoregressive_embedding.shape) if debug else None
        autoregressive_embedding = autoregressive_embedding * tf.cast(select_unit_mask, dtype=tf.float32)

        return units_logits, units, autoregressive_embedding


def test():
    batch_size = 2
    autoregressive_embedding = tf.random.normal((batch_size, AHP.autoregressive_embedding_size))
    action_type = tf.random.uniform(minval=0, maxval=SFS.available_actions, shape=(batch_size, 1), dtype=tf.int32)
    entity_embeddings = tf.random.normal((batch_size, AHP.max_entities, AHP.entity_embedding_size))

    selected_units_head = SelectedUnitsHead()

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    units_logits, units, autoregressive_embedding = \
        selected_units_head.forward(
            autoregressive_embedding, action_type, entity_embeddings)

    if units_logits is not None:
        print("units_logits:", units_logits) if debug else None
        print("units_logits.shape:", units_logits.shape) if debug else None
    else:
        print("units_logits is None!")

    if units is not None:
        print("units:", units) if debug else None
        print("units.shape:", units.shape) if debug else None
    else:
        print("units is None!")

    print("autoregressive_embedding:",
          autoregressive_embedding) if debug else None
    print("autoregressive_embedding.shape:",
          autoregressive_embedding.shape) if debug else None

    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
