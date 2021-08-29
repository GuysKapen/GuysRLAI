import tensorflow as tf

from tensorflow_dl.mini_alpha_star.core.rl.state import MsState

from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS
from tensorflow_dl.mini_alpha_star.libs.hyper_params import ScalarFeature

debug = True


class Feature(object):
    def __init__(self):
        pass

    @staticmethod
    def state2feature(state):
        """

        :param state: MsState
        :return: tensor b x feature_embedding_size (concat state into feature)
        """

        map_data = state.map_state
        batch_entities_tensor = state.entity_state
        scalar_list = state.statistical_state

        batch_size = map_data.shape[0]
        bbo_index = ScalarFeature.beginning_build_order
        scalar_list[bbo_index] = tf.reshape(scalar_list[bbo_index], shape=(batch_size, SFS[bbo_index]))
        for z in scalar_list:
            print(f"z.shape: {z.shape}") if debug else None

        feature_1 = tf.concat(scalar_list, axis=1)
        print(f"feature_1.shape: {feature_1.shape}") if debug else None

        feature_2 = tf.reshape(batch_entities_tensor, shape=(batch_size, AHP.max_entities * AHP.embedding_size))
        print(f"feature_2.shape: {feature_2.shape}") if debug else None

        feature_3 = tf.reshape(map_data, shape=(batch_size, AHP.map_channels * AHP.minimap_size ** 2))
        print(f"feature_3.shape: {feature_3.shape}") if debug else None

        feature = tf.concat([feature_1, feature_2, feature_3], axis=1)
        return feature

    @staticmethod
    def get_size():
        size_all = 0
        for i in ScalarFeature:
            size_all += SFS[i]
        feature_1_size = size_all

        feature_2_size = AHP.max_entities * AHP.embedding_size
        feature_3_size = AHP.map_channels * AHP.minimap_size ** 2
        return feature_1_size + feature_2_size + feature_3_size

    @staticmethod
    def feature2state(feature):
        """
        Split feature into scalar, statistical and map feature and convert to state
        :param feature: b x feature_embedding_size
        :return: MsState
        """

        batch_size = feature.shape[0]

        size_all = 0
        for i in ScalarFeature:
            size_all += SFS[i]

        feature_1_size = size_all
        feature_2_size = AHP.max_entities * AHP.embedding_size
        feature_3_size = AHP.map_channels * AHP.minimap_size ** 2

        print(
            f"feature_1_size + feature_2_size + feature_3_size: {feature_1_size + feature_2_size + feature_3_size}") if debug else None
        assert feature_1_size + feature_2_size + feature_3_size == feature.shape[1]

        feature_1 = feature[:, :feature_1_size]
        scalar_list = []
        last_index = 0

        for i in ScalarFeature:
            scalar_feature = feature_1[:, last_index:last_index + SFS[i]]
            print(f'added scalar_feature.shape: {scalar_feature.shape}') if debug else None
            scalar_list.append(scalar_feature)
            last_index += SFS[i]

        bbo_index = ScalarFeature.beginning_build_order

        print(f'batch_size: {batch_size}') if debug else None
        print(f'scalar_list[bbo_index].shape: {scalar_list[bbo_index].shape}') if debug else None
        scalar_list[bbo_index] = tf.reshape(scalar_list[bbo_index],
                                            shape=(batch_size, SCHP.count_beginning_build_order,
                                                   int(SFS[bbo_index] / SCHP.count_beginning_build_order)))

        feature_2 = feature[:, feature_1_size:feature_1_size + feature_2_size]
        batch_entities = tf.reshape(feature_2, shape=(batch_size, AHP.max_entities, AHP.embedding_size))

        print("feature[:, -feature_3_size:].shape:", feature[:, -feature_3_size:].shape) if debug else None
        print("feature[:, feature_1_size + feature_2_size:].shape:",
              feature[:, feature_1_size + feature_2_size:].shape) if debug else None

        feature_3 = feature[:, -feature_3_size:]
        map_data = tf.reshape(feature_3, shape=(batch_size, AHP.map_channels, AHP.minimap_size, AHP.minimap_size))

        state = MsState(batch_entities, statistical_state=scalar_list, map_state=map_data)

        return state
