import sys
import time
import numpy as np

import tensorflow as tf


class RewardTracker:
    def __init__(self, writer, stop_reward, group_rewards=1):
        self.writer = writer
        self.stop_reward = stop_reward
        self.reward_buf = []
        self.steps_buf = []
        self.group_rewards = group_rewards

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        self.total_steps = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward_steps, frame, epsilon=None):
        reward, steps = reward_steps
        self.reward_buf.append(reward)
        self.steps_buf.append(steps)
        if len(self.reward_buf) < self.group_rewards:
            return False
        reward = np.mean(self.reward_buf)
        steps = np.mean(self.steps_buf)
        self.reward_buf.clear()
        self.steps_buf.clear()
        self.total_rewards.append(reward)
        self.total_steps.append(steps)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        mean_steps = np.mean(self.total_steps[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, mean steps %.2f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards) * self.group_rewards, mean_reward.data, mean_steps.data, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("steps_100", mean_steps, frame)
        self.writer.add_scalar("steps", steps, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_values_of_states(states, net):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = tf.convert_to_tensor(batch)
        action_values_v = net(states_v)
        # best_action_values_v = action_values_v.max(1)[0]
        best_action_values_v = tf.reduce_max(action_values_v, axis=1)[0]
        # mean_vals.append(best_action_values_v.mean().item())
        mean_vals.append(tf.reduce_mean(best_action_values_v).numpy())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)  # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, net, tgt_net, gamma):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = tf.convert_to_tensor(states)
    next_states_v = tf.convert_to_tensor(next_states)
    actions_v = tf.convert_to_tensor(actions)
    rewards_v = tf.convert_to_tensor(rewards)
    done_mask = tf.convert_to_tensor(dones, dtype=tf.bool)
    out = net(states_v)  # (32, 6)
    actions_v = tf.expand_dims(actions_v, -1)
    idx = tf.stack([tf.range(tf.shape(actions_v)[0], dtype=actions_v.dtype), actions_v[:, 0]], axis=-1)  # shape (32, 2)
    state_action_values = tf.gather_nd(out, idx)  # (32,)
    next_state_values = tf.reduce_max(tgt_net(next_states_v), axis=1)
    tf.boolean_mask(next_state_values, done_mask)

    expected_state_action_values = next_state_values * gamma + rewards_v
    return tf.keras.losses.MSE(expected_state_action_values, state_action_values)
