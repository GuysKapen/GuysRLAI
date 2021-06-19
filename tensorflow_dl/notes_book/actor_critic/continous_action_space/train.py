import math
import os
import time

import pybullet_envs
import gym
import numpy as np
import tensorflow as tf

from tensorflow_dl.libs import experience
from tensorflow_dl.notes_book.actor_critic.continous_action_space.agent import AgentA2C
from tensorflow_dl.notes_book.actor_critic.continous_action_space.model import A2C


def test_net(net, env, count=10):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()

        while True:
            obs_v = tf.convert_to_tensor([obs], dtype=tf.float32)
            mu_v = net(obs_v)[0]
            action = tf.squeeze(mu_v, axis=0).numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_log_prob(mu_v, var_v, actions_v):
    """
    calculate log gaussian
    :param mu_v:
    :param var_v:
    :param actions_v:
    :return:
    """
    p1 = -((mu_v - actions_v) ** 2) / (2 * tf.clip_by_value(var_v, clip_value_min=1e-3, clip_value_max=tf.float32.max) ** 2)
    p2 = -tf.math.log(tf.math.sqrt(2 * var_v ** 2 * math.pi))

    return p1 + p2


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
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = tf.convert_to_tensor(np.array(states, copy=False), dtype=tf.float32)
    actions_v = tf.convert_to_tensor(actions, dtype=tf.float32)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = tf.convert_to_tensor(np.array(last_states, copy=False), dtype=tf.float32)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = tf.convert_to_tensor(rewards_np, dtype=tf.float32)
    return states_v, actions_v, ref_vals_v


ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
CLIP_GRAD = 0.1

TEST_ITERS = 1000

if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    # save_path = "/content/data/MyDrive/models/Robot"
    save_path = "robot"
    if os.path.exists(save_path):
        net = tf.keras.models.load_model(save_path)
        print("#" * 60)
        print("Restored saved model!")
    else:
        net = A2C(env.action_space.shape[0])

    agent = AgentA2C(net)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    writer = tf.summary.create_file_writer("a2c-robot")

    best_reward = 2
    batch = []
    for step_idx, exp in enumerate(exp_source):
        if step_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(net, test_env)
            print("Test done is %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            with writer.as_default():
                tf.summary.scalar("test_reward", rewards, step_idx)
                tf.summary.scalar("test_steps", steps, step_idx)
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)
                    net.save(save_path)
                best_reward = rewards

        batch.append(exp)
        if len(batch) < BATCH_SIZE:
            continue

        states_v, actions_v, vals_ref_v = unpack_batch(batch, net)
        batch.clear()

        with tf.GradientTape() as g:
            mu_v, var_v, value_v = net(states_v)
            loss_value_v = tf.keras.losses.MSE(tf.squeeze(value_v, axis=-1), vals_ref_v)
            adv_v = tf.expand_dims(vals_ref_v, axis=-1) - tf.stop_gradient(value_v)

            log_prob_v = adv_v * calc_log_prob(mu_v, var_v, actions_v)
            loss_policy_v = -tf.reduce_mean(log_prob_v)
            entropy_loss_v = ENTROPY_BETA * tf.reduce_mean(-(tf.math.log(2 * math.pi * var_v) + 1) / 2)

            loss_v = entropy_loss_v + loss_value_v + loss_policy_v

        gradients = g.gradient(loss_v, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        loss_v += loss_policy_v
