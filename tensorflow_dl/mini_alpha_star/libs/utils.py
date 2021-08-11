import tensorflow as tf
import numpy as np


debug = True


def stable_multinomial(probs=None, logits=None, temperature=1, num_samples=1, min_prob=1e-10,
                       max_logit=1e+10, min_temperature=1e-10, max_temperature=1e+10):
    if probs is not None:
        probs = tf.clip_by_value(probs, clip_value_min=min_prob, clip_value_max=np.inf)
        logits = tf.math.log(probs)

    logits = tf.clip_by_value(clip_value_max=max_logit, clip_value_min=-np.inf)
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
    unit_type_name, race = None


def get_unit_type_name_and_race(unit_type):
