import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import nn as F

from tensorflow_dl.mini_alpha_star.core.arch.spatial_encoder import ResBlock1D
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.transformer import Transformer

debug = True


class Baseline(tf.keras.Model):
    """
        Inputs: prev_state, scalar_features, opponent_observations, cumulative_score, action_type, lstm_output
        Outputs:
            winloss_baseline - A baseline value used for the `action_type` argument
    """

    def __init__(self, baseline_type='winloss',
                 n_statistics=10,
                 baseline_input=AHP.winloss_baseline_input_size,
                 n_upgrades=SFS.upgrades,
                 n_units_buildings=SFS.unit_counts_bow,
                 n_effects=SFS.effects, n_upgrade=SFS.upgrade,
                 n_resblocks=AHP.n_resblocks,
                 original_32=AHP.original_32,
                 original_64=AHP.original_64,
                 original_128=AHP.original_128,
                 original_256=AHP.original_256):
        super(Baseline, self).__init__()
        self.baseline_type = baseline_type
        if baseline_type == 'build_order':
            baseline_input = AHP.build_order_baseline_input_size
        elif baseline_type == 'built_units':
            baseline_input = AHP.built_units_baseline_input_size
        elif baseline_type == 'upgrades':
            baseline_input = AHP.upgrades_baseline_input_size
        elif baseline_type == 'effects':
            baseline_input = AHP.effects_baseline_input_size

        self.statistics_fc = layers.Dense(original_64)  # with relu
        self.upgrades_fc = layers.Dense(original_128)  # with relu
        self.unit_counts_bow_fc = layers.Dense(original_64)  # A bag-of-words unit count from `entity_list`, with relu
        self.units_buildings_fc = layers.Dense(original_32)  # with relu, also goto scalar_context
        self.effects_fc = layers.Dense(original_32)  # with relu, also goto scalar_context
        self.upgrade_fc = layers.Dense(
            original_32)  # with relu, also goto scalar_context. What is the difference with upgrades_fc?
        self.before_beginning_build_order = layers.Dense(16)  # without relu
        self.beginning_build_order_transformer = Transformer(d_model=16, d_inner=32,
                                                             n_layers=3, n_head=2, d_k=8, d_v=8, dropout=0.1)
        self.relu = layers.ReLU()

        self.embed_fc = layers.Dense(original_256)  # with relu
        self.resblock_stack = [
            ResBlock1D(inplanes=original_256, planes=original_256, seq_len=1)
            for _ in range(n_resblocks)]

        self.out_fc = layers.Dense(1)

    def preprocess(self, various_observations):
        [agent_statistics, upgrades, unit_counts_bow,
         units_buildings, effects, upgrade, beginning_build_order] = various_observations

        # These features are all concatenated together to yield `action_type_input`,
        # passed through a linear of size 256, then passed through 16 ResBlocks with 256 hidden units
        # and layer normalization, passed through a ReLU, then passed through
        # a linear with 1 hidden unit.

        embedded_scalar_list = []
        # agent_statistics: Embedded by taking log(agent_statistics + 1) and passing through a linear of size 64 and a ReLU
        the_log_statistics = tf.math.log(agent_statistics + 1)
        if tf.reduce_any(tf.math.is_nan(the_log_statistics)).numpy():
            print('Find NAN the_log_statistics !', the_log_statistics)
            eps = 1e-9
            the_log_statistics = tf.math.log(self.relu(agent_statistics + 1) + eps)

            if tf.reduce_any(tf.math.is_nan(the_log_statistics)).numpy():
                print('Find NAN the_log_statistics !', the_log_statistics)
                the_log_statistics = tf.ones_like(agent_statistics)

        x = F.relu(self.statistics_fc(the_log_statistics))
        embedded_scalar_list.append(x)

        # TODO: `cumulative_score`, as a 1D tensor of values, is processed like `agent_statistics`.
        cumulative_score = tf.ones((agent_statistics.shape[0], SFS.agent_statistics))
        score_log_statistics = tf.math.log(cumulative_score + 1)
        x = F.relu(self.statistics_fc(score_log_statistics))
        embedded_scalar_list.append(x)

        # upgrades: The boolean vector of whether an upgrade is present is embedded through a linear of size 128 and
        # a ReLU
        x = F.relu(self.upgrades_fc(upgrades))
        embedded_scalar_list.append(x)

        # unit_counts_bow: A bag-of-words unit count from `entity_list`. The unit count vector is embedded by square
        # rooting, passing through a linear layer, and passing through a ReLU
        x = F.relu(self.unit_counts_bow_fc(unit_counts_bow))
        embedded_scalar_list.append(x)

        # cumulative_statistics: The cumulative statistics (including units, buildings, effects, and upgrades) are preprocessed
        # into a boolean vector of whether or not statistic is present in a human game.
        # That vector is split into 3 sub-vectors of units/buildings, effects, and upgrades,
        # and each subvector is passed through a linear of size 32 and a ReLU, and concatenated together.
        # The embedding is also added to `scalar_context`

        cumulative_statistics = []  # it is different in different baseline
        if self.baseline_type == "winloss" or self.baseline_type == "build_order" or self.baseline_type == "built_units":
            x = F.relu(self.units_buildings_fc(units_buildings))
            cumulative_statistics.append(x)
        if self.baseline_type == "effects" or self.baseline_type == "build_order":
            x = F.relu(self.effects_fc(effects))
            cumulative_statistics.append(x)
        if self.baseline_type == "upgrades" or self.baseline_type == "build_order":
            x = F.relu(self.upgrade_fc(upgrade))
            cumulative_statistics.append(x)
        embedded_scalar_list.extend(cumulative_statistics)

        # beginning_build_order: The first 20 constructed entities are converted to a 2D tensor of size
        # [20, num_entity_types], concatenated with indices and the binary encodings
        # (as in the Entity Encoder) of where entities were constructed (if applicable).
        # The concatenation is passed through a transformer similar to the one in the entity encoder,
        # but with keys, queries, and values of 8 and with a MLP hidden size of 32.
        # The embedding is also added to `scalar_context`.
        print("beginning_build_order:", beginning_build_order) if debug else None
        print("beginning_build_order.shape:", beginning_build_order.shape) if debug else None

        x = self.beginning_build_order_transformer(self.before_beginning_build_order(beginning_build_order))
        print("x:", x) if debug else None
        print("x.shape:", x.shape) if debug else None

        x = tf.reshape(x, shape=(x.shape[0], SCHP.count_beginning_build_order * 16))
        print("x:", x) if debug else None
        print("x.shape:", x.shape) if debug else None
        embedded_scalar_list.append(x)

        embedded_scalar = tf.concat(embedded_scalar_list, axis=1)

        print("self.baseline_type:", self.baseline_type) if debug else None
        print("embedded_scalar.shape:", embedded_scalar.shape) if debug else None

        return embedded_scalar

    def forward(self, lstm_output, various_observations, opponent_observations):
        player_scalar_out = self.preprocess(various_observations)
        # AlphaStar: The baseline extracts those same observations from `opponent_observations`.
        opponenet_scalar_out = self.preprocess(opponent_observations)

        # AlphaStar: These features are all concatenated together to yield `action_type_input`
        action_type_input = tf.concat([lstm_output, player_scalar_out, opponenet_scalar_out], axis=1)
        print("action_type_input.shape:", action_type_input.shape) if debug else None

        # AlphaStar: passed through a linear of size 256
        x = self.embed_fc(action_type_input)
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: then passed through 16 ResBlocks with 256 hidden units and layer normalization,
        x = tf.expand_dims(x, axis=-1)
        x = tf.transpose(x, perm=[0, 2, 1])
        print("before resblock x.shape: ", x.shape) if debug else None
        for resblock in self.resblock_stack:
            x = resblock(x)
        print("after resblock x.shape: ", x.shape) if debug else None
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.squeeze(x, axis=-1)
        print("x.shape:", x.shape) if debug else None

        # AlphaStar: passed through a ReLU
        x = F.relu(x)

        # AlphaStar: then passed through a linear with 1 hidden unit.
        baseline = self.out_fc(x)
        print("baseline:", baseline) if debug else None

        # AlphaStar: This baseline value is transformed by ((2.0 / PI) * atan((PI / 2.0) * baseline)) and is used as
        # the baseline value
        out = (2.0 / np.pi) * tf.math.atan((np.pi / 2.0) * baseline)
        print("out:", out) if debug else None

        return out

    def call(self, inputs, training=None, mask=None):
        lstm_output, various_observations, opponent_observations = inputs
        return self.forward(lstm_output, various_observations, opponent_observations)


def test():
    base_line = Baseline()

    batch_size = 2
    # dummy scalar list
    scalar_list = []

    agent_statistics = tf.ones((batch_size, SFS.agent_statistics))
    upgrades = tf.random.normal((batch_size, SFS.upgrades))
    unit_counts_bow = tf.random.normal((batch_size, SFS.unit_counts_bow))
    units_buildings = tf.random.normal((batch_size, SFS.units_buildings))
    effects = tf.random.normal((batch_size, SFS.effects))
    upgrade = tf.random.normal((batch_size, SFS.upgrade))
    beginning_build_order = tf.random.normal((batch_size, SCHP.count_beginning_build_order,
                                              int(SFS.beginning_build_order / SCHP.count_beginning_build_order)))

    scalar_list.append(agent_statistics)
    scalar_list.append(upgrades)
    scalar_list.append(unit_counts_bow)
    scalar_list.append(units_buildings)
    scalar_list.append(effects)
    scalar_list.append(upgrade)
    scalar_list.append(beginning_build_order)

    opponenet_scalar_out = scalar_list

    lstm_output = tf.ones((batch_size, AHP.lstm_hidden_dim))

    out = base_line.forward(lstm_output, scalar_list, opponenet_scalar_out)

    print("out:", out) if debug else None
    print("out.shape:", out.shape) if debug else None

    if debug:
        print("This is a test!")


if __name__ == '__main__':
    test()
