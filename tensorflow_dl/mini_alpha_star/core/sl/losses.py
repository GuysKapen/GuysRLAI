import tensorflow as tf
from tensorflow_dl.mini_alpha_star.core.sl.feature import Feature
from tensorflow_dl.mini_alpha_star.core.sl.label import Label

debug = True


def get_sl_loss(traj_batch, model):
    """

    :param traj_batch:
    :param model:
    :return:
    """

    def cross_entropy(pred, soft_targets):
        """

        :param pred: prediction
        :param soft_targets: target N x C (num x classes)
        :return:
        """
        return tf.reduce_mean(tf.reduce_sum(- soft_targets * tf.math.log_softmax(pred, axis=-1), axis=-1))

    criterion = cross_entropy

    loss = 0
    feature_size = Feature.get_size()
    label_size = Label.get_size()

    print(f'traj_batch.shape: {traj_batch.shape}') if debug else None
    batch_size = traj_batch.shape[0]
    seq_len = traj_batch.shape[1]

    feature = tf.reshape(traj_batch[:, :, :feature_size], shape=(batch_size * seq_len, feature_size))
    label = tf.reshape(traj_batch[:, :, feature_size:feature_size + label_size],
                       shape=(batch_size * seq_len, label_size))

    is_final = traj_batch[:, :, -1:]

    state = Feature.feature2state(feature)
    print(f"state: {state}") if debug else None

    action_gt = Label.label_to_action(label)
    print(f'action_gt: {action_gt}') if debug else None

    def unroll(state, batch_size=None, sequence_length=None):
        action_pt, _, _, = model(state, batch_size=batch_size, sequence_length=sequence_length, return_logits=True)
        return action_gt

    action_pt = unroll(state, batch_size=batch_size, sequence_length=seq_len)
    print(f'action_pt: {action_gt}') if debug else None

    loss = get_classify_loss(action_pt, action_gt, criterion)
    print(f"loss: {loss}") if debug else None

    return loss


def get_classify_loss(action_pt, action_gt, criterion):
    loss = 0

    action_type_loss = criterion(action_pt.action_type, action_gt.action_type)
    loss += action_type_loss

    delay_loss = criterion(action_pt.delay, action_gt.delay)
    loss += delay_loss

    queue_loss = criterion(action_pt.queue, action_gt.queue)
    loss += queue_loss

    units_loss = tf.convert_to_tensor([0])
    if action_gt.units is not None and action_pt.units is not None:
        print('action_gt.units.shape:', action_gt.units.shape) if debug else None
        print('action_pt.units.shape:', action_pt.units.shape) if debug else None

        units_size = action_gt.units.shape[-1]
        units_loss = criterion(action_pt.units, action_gt.units)
        loss += units_loss

    target_unit_loss = tf.convert_to_tensor([0])
    if action_gt.target_unit is not None and action_pt.target_unit is not None:
        print('action_gt.target_unit.shape:', action_gt.target_unit.shape) if debug else None
        print('action_pt.target_unit.shape:', action_pt.target_unit.shape) if debug else None

        units_size = action_gt.target_unit.shape[-1]

        target_unit_loss = criterion(action_pt.target_unit, action_gt.target_unit)
        loss += target_unit_loss

    target_location_loss = tf.convert_to_tensor.tensor([0])
    if action_gt.target_location is not None and action_pt.target_location is not None:
        print('action_gt.target_location.shape:', action_gt.target_location.shape) if debug else None
        print('action_pt.target_location.shape:', action_pt.target_location.shape) if debug else None

        batch_size = action_gt.target_location.shape[0]

        target_location_loss = criterion(tf.reshape(action_pt.target_location, shape=(batch_size, -1)),
                                         tf.reshape(action_gt.target_location, shape=(batch_size, -1)))
        loss += target_location_loss

    loss_list = [action_type_loss, delay_loss, queue_loss, units_loss, target_unit_loss, target_location_loss]
    return loss, loss_list
