import tensorflow as tf

from tensorflow_dl.mini_alpha_star.core.rl.action import ArgsActionLogits
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import LabelIndex
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Label_Size as LS
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP

debug = True


class Label(object):
    def __init__(self):
        super(Label, self).__init__()

    @staticmethod
    def action_list_to_label(action_list):
        """

        :param action_list: args action list
        :return: tensor b x label_feature_size
        """

        tle_index = LabelIndex.target_location_encoding
        target_location_encoding = action_list[tle_index]

        batch_size = target_location_encoding.shape[0]

        print(f"target_location_encoding.shape before: {target_location_encoding.shape}") if debug else None
        target_location_encoding = tf.reshape(target_location_encoding, shape=(batch_size, LS[tle_index]))
        print(f"target_location_encoding.shape after: {target_location_encoding.shape}") if debug else None

        action_list[tle_index] = target_location_encoding
        label = tf.concat(action_list, axis=1)
        return label

    @staticmethod
    def get_size():
        return sum([LS[i] for i in LabelIndex])

    @staticmethod
    def action_to_label(action):
        """

        :param action: args action logits
        :return: tensor b x label_feature_size
        """
        target_location_index = LabelIndex.target_location_encoding
        target_location_encoding = action.target_location
        batch_size = target_location_encoding.shape[0]
        print("target_location_encoding.shape before:", target_location_encoding.shape) if debug else None
        target_location_encoding = target_location_encoding.reshape(batch_size, LS[target_location_index])
        print("target_location_encoding.shape after:", target_location_encoding.shape) if debug else None
        action.target_location = target_location_encoding

        units_index = LabelIndex.select_units_encoding
        units_encoding = action.units
        units_encoding = tf.reshape(units_encoding, shape=(batch_size, LS[units_index]))
        action.units = units_encoding

        target_unit_index = LabelIndex.target_unit_encoding
        target_unit_encoding = action.target_unit
        target_unit_encoding = tf.reshape(target_unit_encoding, shape=(batch_size, LS[target_unit_index]))
        action.target_unit = target_unit_encoding

        label = tf.concat(action.to_list(), axis=1)
        return label

    @staticmethod
    def label_to_action(label):
        """

        :param label: b x label_feature_size
        :return: args action logits tensor
        """

        batch_size = label.shape[0]
        action = None

        action_list = []
        last_index = 0
        for i in LabelIndex:
            label_i = label[:, last_index:last_index + LS[i]]
            print(f'added label_i.shape: {label_i.shape}') if debug else None
            action_list.append(action)
            last_index += LS[i]

        tle_index = LabelIndex.target_location_encoding
        print("action_list[tue_index].shape before:", action_list[tle_index].shape) if debug else None
        action_list[tle_index] = tf.reshape(action_list[tle_index], shape=(batch_size, SCHP.world_size,
                                                                           int(LS[tle_index] / SCHP.world_size)))
        print("action_list[tue_index].shape before:", action_list[tle_index].shape) if debug else None

        units_index = LabelIndex.select_units_encoding
        action_list[units_index] = tf.reshape(action_list[units_index], shape=(
            batch_size, AHP.max_selected, int(LS[units_index] / AHP.max_selected)))

        target_unit_index = LabelIndex.target_unit_encoding
        action_list[target_unit_index] = tf.reshape(action_list[target_unit_index],
                                                    shape=(batch_size, int(LS[target_unit_index])))

        return ArgsActionLogits(*action_list)

    @staticmethod
    def label_to_action_list(label):
        """

        :param label: b x label_feature_size
        :return: args action list
        """
        batch_size = label.shape[0]
        action = None

        action_list = []
        last_index = 0
        for i in LabelIndex:
            label_i = label[:, last_index:last_index + LS[i]]
            print(f'added label_i.shape: {label_i.shape}') if debug else None
            action_list.append(action)
            last_index += LS[i]

        tle_index = LabelIndex.target_location_encoding
        print("action_list[tue_index].shape before:", action_list[tle_index].shape) if debug else None
        action_list[tle_index] = tf.reshape(action_list[tle_index], shape=(batch_size, SCHP.world_size,
                                                                           int(LS[tle_index] / SCHP.world_size)))
        print("action_list[tue_index].shape before:", action_list[tle_index].shape) if debug else None

        return action_list
