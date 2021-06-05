import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.libs import common, experience, actions, agent

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class AtariA2C(tf.keras.Model):
    def __init__(self, n_actions):
        super(AtariA2C, self).__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            layers.Flatten()
        ])

        self.policy = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(n_actions)
        ])

        self.value = tf.keras.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, x, training=None, mask=None):
        fx = tf.cast(x, tf.float32) / 256
        conv_out = self.conv(fx)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = tf.convert_to_tensor(np.array(states, copy=False))
    actions_t = tf.convert_to_tensor(actions, tf.int64)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = tf.convert_to_tensor(np.array(last_states, copy=False), tf.float32)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = tf.convert_to_tensor(rewards_np, tf.float32)
    return states_v, actions_t, ref_vals_v


if __name__ == '__main__':
    make_env = lambda: common.wrap_dqn(gym.make('PongNoFrameskip-v4'))
    envs = [make_env() for _ in range(NUM_ENVS)]

    net = AtariA2C(envs[0].action_space.n)

    agent = agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True)
    exp_source = experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-3)

    batch = []

    for step_idx, exp in enumerate(exp_source):
        batch.append(exp)

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            print("New rewards: ", new_rewards)

        if len(batch) < BATCH_SIZE:
            continue

        states_v, actions_t, vals_ref_v = unpack_batch(batch, net)
        batch.clear()

        with tf.GradientTape() as g:
            logits_v, value_v = net(states_v)
            loss_value_v = tf.keras.losses.MSE(tf.squeeze(value_v, axis=-1), vals_ref_v)

            log_prob_v = tf.nn.log_softmax(logits_v, axis=1)
            tf.stop_gradient(value_v)
            adv_v = vals_ref_v - tf.squeeze(value_v, axis=-1)
            # log_prob_actions_v = adv_v * log_prob_v[list(range(BATCH_SIZE)), actions_t]
            log_prob_actions_v = tf.expand_dims(adv_v, axis=-1) * tf.gather(log_prob_v, actions_t)[:BATCH_SIZE]
            loss_policy_v = -tf.reduce_mean(log_prob_actions_v)

            prob_v = tf.nn.softmax(logits_v, axis=1)
            entropy_loss_v = tf.reduce_mean(tf.reduce_sum(ENTROPY_BETA * (prob_v * log_prob_v)))

            loss_v = entropy_loss_v + loss_value_v

        gradients = g.gradient(loss_v, net.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, CLIP_GRAD)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        loss_v += loss_policy_v
