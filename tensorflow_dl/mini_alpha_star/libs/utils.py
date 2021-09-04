import os

import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_dl.mini_alpha_star.libs.hyper_params import StarCraft_Hyper_Parameters as SCHP
from tensorflow_dl.mini_alpha_star.libs.hyper_params import Scalar_Feature_Size as SFS
from pysc2.lib import actions
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg

debug = True


def stable_multinomial(probs=None, logits=None, temperature=1, num_samples=1, min_prob=1e-10,
                       max_logit=1e+10, min_temperature=1e-10, max_temperature=1e+10):
    if probs is not None:
        probs = tf.clip_by_value(probs, clip_value_min=min_prob, clip_value_max=np.inf)
        logits = tf.math.log(probs)

    logits = tf.clip_by_value(logits, clip_value_max=max_logit, clip_value_min=-np.inf)
    temperature = np.clip(temperature, min_temperature, max_temperature)
    logits = (logits - tf.reduce_max(logits)) / temperature
    probs = tf.exp(logits)
    print('probs:', probs) if debug else None
    print('max probs:', probs.max()) if debug else None
    print('min probs:', probs.min()) if debug else None

    return tf.random.categorical(probs, num_samples)


def unit_type_to_unit_type_idx(unit_type):
    """
    Transform unique unit type in sc2 to unit index in one hot
    :param unit_type:
    :return:
    """
    unit_type_name, race = get_unit_type_name_and_race(unit_type)
    print('unit_type_name, race: ', unit_type_name, race) if debug else None

    unit_type_index = get_unit_type_index(unit_type_name, race)
    print('unit_type_index: ', unit_type_index) if debug else None

    return unit_type_index


def get_unit_type_name_and_race(unit_type):
    for race in (Neutral, Protoss, Terran, Zerg):
        try:
            return race(unit_type), race
        except ValueError:
            pass


def get_unit_type_index(unit_type_name, race):
    begin_index = 0
    if race == Neutral:
        begin_index = 0
    elif race == Protoss:
        begin_index = len(Neutral)
    elif race == Terran:
        begin_index = len(Protoss) + len(Neutral)
    elif race == Zerg:
        begin_index = len(Terran) + len(Neutral) + len(Protoss)

    for i, e in enumerate(list(race)):
        if e == unit_type_name:
            return i + begin_index
    return -1


def unpack_bits_for_large_number(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError('numpy data type need to be int-like')
    x_shape = list(x.shape)
    x = np.reshape(x, (-1, 1))
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape((1, num_bits))
    return (x & mask).astype(bool).astype(int).reshape(x_shape + [num_bits])


def calculate_unit_counts_bow(obs):
    unit_counts = obs["unit_counts"]
    print('unit_counts: ', unit_counts) if debug else None
    unit_counts_bow = tf.zeros(1, SFS.unit_counts_bow)
    for u_c in unit_counts:
        unit_type = u_c[0]
        unit_count = u_c[1]
        # unit can not be negative
        assert unit_type >= 0
        assert unit_count >= 0

        # the unit_type should not  be more than the SFS.unit_counts_bow
        # if it is, make it to be 0 now. (0 means nothing now)
        # the most impact one is ShieldBattery = 1910
        # find a better way to do it: transform into unit_type_index
        unit_type_index = unit_type_to_unit_type_idx(unit_type)
        if unit_type_index >= SFS.unit_counts_bow:
            unit_type_index = 0

        unit_counts_bow[0, unit_type_index] = unit_count

    return unit_counts_bow


def load_latest_model(model_type, path):
    models = list(filter(lambda x: model_type in x, os.listdir(path)))
    if len(models) == 0:
        print("No models found")
        return None

    models.sort()
    model_path = os.path.join(path, models[-1])
    print(f"Load model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def load_the_model(model_path):
    return tf.keras.models.load_model(model_path)


def calculate_build_order(previous_bo, obs, next_obs):
    ucb = calculate_unit_counts_bow(obs)
    next_ucb = calculate_unit_counts_bow(next_obs)
    diff = next_ucb - ucb

    # the probe, drone, and SCV are not counted in build order
    worker_type_list = [84, 104, 45]
    # the pylon, drone, and supplypot are not counted in build order
    supply_type_list = [60, 106, 19]
    diff[0, worker_type_list] = 0
    diff[0, supply_type_list] = 0

    diff_count = tf.reduce_sum(diff)
    print("Diff between unit_counts_bow", diff_count) if debug else None
    if diff_count == 1.0:
        diff_numpy = diff.numpy()
        index_list = np.asarray(diff_numpy >= 1.0).nonzero()
        print("index_list: ", index_list) if debug else None
        index = index_list[1][0]
        if index not in worker_type_list and index not in supply_type_list:
            previous_bo.append(index)

    return previous_bo


def show_map_data_test(obs, map_width=128, show_original=True, show_resacel=True):
    use_small_map = False
    small_map_width = 32

    resize_type = np.uint8
    save_type = np.float16

    # note in pysc2-1.2 obs["feature_minimap"]["height_map"] can be shown straight,
    # however in pysc-3.0, that can not be show straight, must be transformed to numpy array firstly
    height_map = np.array(obs["feature_minima;"]['height_map'])
    if show_original:
        plt.imshow(height_map)
        plt.show()

    visibility_map = np.array(obs["feature_minimap"]["visibility_map"])
    if show_original:
        plt.imshow(visibility_map)
        plt.show()

    creep = np.array(obs["feature_minimap"]["creep"])
    if show_original:
        plt.imshow(creep)
        plt.show()

    player_relative = np.array(obs["feature_minimap"]["player_relative"])
    if show_original:
        plt.imshow(player_relative)
        plt.show()

    # the below three maps are all zero, this may due to we connect to a 3.16.1 version pysc2
    # may be different when we connect to 4.10 version sc2
    alerts = np.array(obs["feature_minimap"]["alerts"])
    if show_original:
        plt.imshow(alerts)
        plt.show()

    pathable = np.array(obs["feature_minimap"]["pathable"])
    if show_original:
        plt.imshow(pathable)
        plt.show()

    buildable = np.array(obs["feature_minimap"]["buildable"])
    if show_original:
        plt.imshow(buildable)
        plt.show()

    return None


def show_numpy_image(numpy_image):
    """

    :param numpy_image:
    :return:
    """
    plt.imshow(numpy_image)
    plt.show()
    return None


def np_one_hot(targets, nb_classes):
    """
    This is for numpy array
    :param targets:
    :param nb_classes:
    :return:
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form
    :param labels: Long tensor class labels, sized (N,)
    :param num_classes: int number of classes
    :return: tensor encoded labels, sized (N, #classes)
    """
    y = tf.eye(num_classes)

    return tf.gather(y, labels)


def to_one_hot(y, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims and convert it to 1-hot representation
    with n+1 dims
    :param y:
    :param n_dims:
    :return:
    """
    y_tensor = tf.reshape(tf.cast(y, dtype=tf.int64), shape=(-1, 1))
    n_dims = n_dims if n_dims is not None else int(tf.reduce_max(y_tensor)) + 1

    return tf.one_hot(y_tensor, depth=n_dims)


def action_can_be_queued(action_type):
    """
    Test the action type whether can be queued
    :param action_type: int
    :return:
    """
    need_args = actions.RAW_FUNCTIONS[action_type].args
    result = False
    for arg in need_args:
        if arg.name == "queued":
            result = True
            break
    return result


def action_can_be_queued_mask(action_types):
    """

    :param action_types:
    :return:
    """
    mask = np.zeros_like(action_types)
    action_types = tf.stop_gradient(action_types).numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()
        print(f'i: {i}, action_type_index: {action_type_index}') if debug else None
        mask[i] = action_can_be_queued(action_type_index)

    return mask


def action_can_apply_to_entity_types(action_type):
    """
    Find the entity types which the action_type can be applied to
    :param action_type:
    :return: mask of applied entity types
    """
    mask = tf.ones(shape=(1, SCHP.max_unit_type))

    # note this can be done when know which action_type can aplly to certain unit_types
    # which need strong prior knowledge
    # At present there is no apt in pysc2, thus now we only return a mask means all unit_types accept the action_type
    return mask


def action_can_apply_to_entity_types_mask(action_types):
    """
    Find the entity_types which the action-type can be applied to
    :param action_types:
    :return: mask
    """
    mask_list = []
    action_types = tf.stop_gradient(action_types).numpy()

    for i, action_type in enumerate(action_types):
        print(f'i: {i}, action_type: {action_type}') if debug else None
        mask = action_can_apply_to_entity_types(action_type)
        mask_list.append(mask)

    batch_mask = tf.concat(mask_list, axis=0)

    return batch_mask


def action_can_apply_to_entity(action_type):
    """
    Find the entity_types which the action_type can be applied to
    :param action_type:
    :return: list of applied entity_types
    """
    if action_type % 2 == 0:
        return [0, 2, 4]
    else:
        return [1, 3, 7, 11]


def action_involve_selection_units(action_type):
    """
    Check the action_type whether involve selecting units
    :param action_type:
    :return:
    """

    need_args = actions.RAW_FUNCTIONS[action_type].args
    result = False
    for arg in need_args:
        if arg.name == 'unit_tags':
            result = True
            break
    return result


def action_involve_selecting_units_mask(action_types):
    """
    Check the action_type whether involve selecting units
    :param action_types:
    :return: mask
    """

    mask = tf.zeros_like(action_types)
    action_types = tf.stop_gradient(action_types).numpy()

    for i, action_type in enumerate(action_types):
        print(f'i: {i}, action_type: {action_type}') if debug else None

        mask[i] = action_involve_selection_units(action_type)

    return mask


def action_involve_targeting_units(action_type):
    """
    Check the action_type whether involve targeting units
    :param action_type:
    :return:
    """
    need_args = actions.RAW_FUNCTIONS[action_type].args
    for arg in need_args:
        if arg.name == 'target_unit_tag':
            return True
    return False


def action_involve_targeting_units_mask(action_types):
    """
    Check the action_type whether involve targeting units
    :param action_types:
    :return:
    """

    mask = tf.zeros_like(action_types)
    action_types = tf.stop_gradient(action_types).numpy()

    for i, action_type in enumerate(action_types):
        print(f'i: {i}, aciton_type: {action_types}') if debug else None

        mask[i] = action_involve_targeting_units(action_type)

    return mask


def action_involve_targeting_location(action_type):
    """
    Check the action_type whether involve targeting location
    :param action_type:
    :return:
    """
    need_args = actions.RAW_FUNCTIONS[action_type].args
    for arg in need_args:
        if arg.name == 'world':
            return True
    return False


def action_involve_targeting_location_mask(action_types):
    """
    Check the action_type whether involve targeting location
    :param action_types:
    :return: mask
    """

    mask = np.zeros_like(action_types)
    action_types = tf.stop_gradient(action_types).numpy()

    for i, action_type in enumerate(action_types):
        action_type_index = action_type.item()

        print('i:', i, 'action_type_index:', action_type_index) if debug else None

        mask[i] = action_involve_targeting_location(action_type_index)

    return mask
