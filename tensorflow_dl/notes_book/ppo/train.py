import math
import os
import time

import pybullet_envs
import gym
import numpy as np
from tensorflow_dl.libs import experience
from tensorflow_dl.notes_book.actor_critic.continous_action_space.model import *
from tensorflow_dl.notes_book.ppo.model import Actor, Critic, AgentA2C

ENV_ID = "HalfCheetahBulletEnv-v0"
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-3

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 100000


def test_net(net, env, count=10):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()

        while True:
            obs_v = tf.convert_to_tensor([obs], dtype=tf.float32)
            mu_v = net(obs_v)
            action = tf.squeeze(mu_v, axis=0).numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_log_prob(mu_v, logstd_v, actions_v):
    """
    calculate log gaussian
    :param mu_v:
    :param logstd_v: log of standard
    :param actions_v:
    :return:
    """
    p1 = -((mu_v - actions_v) ** 2) / (
            2 * tf.clip_by_value(tf.math.exp(logstd_v), clip_value_min=1e-3, clip_value_max=tf.float32.max) ** 2)
    p2 = -tf.math.log(tf.math.sqrt(tf.math.exp(logstd_v) * 2 * math.pi))

    return p1 + p2


def calc_adv_ref(trajectory, net_crt, states_v):
    """
    Calculate advantage and 1 step ref value
    :param trajectory: trajectory list
    :param net_crt: critic net
    :param states_v: states net
    :return: tuple with advantage numpy array and reference values
    """

    values_v = net_crt(states_v)
    values = tf.squeeze(values_v).numpy()

    last_gae = 0.0
    result_adv = []
    result_ref = []

    for val, next_val, (exp,) in zip(reversed(values[:-1]), reversed(values[1:]), trajectory[:-1]):
        # calculate reversed using trick from formula:
        # advantage = delta + (y * gamma) * delta_(t+1) + (y * gamma)^2 * delta_(t+1) + ... + (y * gamma)^(T-t+1) * delta_(t+1)
        # and delta = reward + y * value_t+1 - value_t
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae

        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv = tf.convert_to_tensor(list(reversed(result_adv)), dtype=tf.float32)
    ref = tf.convert_to_tensor(list(reversed(result_ref)), dtype=tf.float32)

    return adv, ref


if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    # save_path = "/content/data/MyDrive/models/Robot"
    save_path = "robot-ppo"
    if os.path.exists(save_path):
        net = tf.keras.models.load_model(save_path)
        print("#" * 60)
        print("Restored saved model!")
    else:
        net = A2C(env.action_space.shape[0])

    net_act = Actor(env.action_space.shape[0])
    net_crt = Critic(env.observation_space.shape[0])
    agent = AgentA2C(net_act)
    exp_source = experience.ExperienceSource(env, agent, steps_count=1)
    act_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ACTOR)
    crt_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_CRITIC)
    writer = tf.summary.create_file_writer("a2c-robot")

    trajectory = []
    best_reward = None

    for step_idx, exp in enumerate(exp_source):
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            reward, steps = zip(*rewards_steps)
            with writer.as_default():
                tf.summary.scalar("episode_steps", np.mean(steps), step_idx)

        if step_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(net_act, test_env)

            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    net_act.save(save_path)
                best_reward = rewards

        trajectory.append(exp)
        if len(trajectory) < TRAJECTORY_SIZE:
            continue

        traj_states = [t[0].state for t in trajectory]
        traj_actions = [t[0].action for t in trajectory]
        traj_states_v = tf.convert_to_tensor(traj_states)
        traj_actions_v = tf.convert_to_tensor(traj_actions)
        traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v)
        mu_v = net_act(traj_states_v)

        old_log_prob_v = calc_log_prob(mu_v, net_act.logstd, traj_actions_v)

        # Normalize
        traj_adv_v = (traj_adv_v - tf.reduce_mean(traj_adv_v)) / tf.math.reduce_std(traj_adv_v)

        trajectory = trajectory[:-1]
        old_log_prob_v = tf.stop_gradient(old_log_prob_v[:-1])

        sum_loss_value = sum_loss_policy = count_steps = 0.0

        for epoch in range(PPO_EPOCHES):
            for batch_offset in range(0, len(trajectory), PPO_BATCH_SIZE):
                states_v = traj_states_v[batch_offset:batch_offset + PPO_BATCH_SIZE]
                actions_v = traj_actions[batch_offset:batch_offset + PPO_BATCH_SIZE]

                batch_adv_v = tf.expand_dims(traj_adv_v[batch_offset:batch_offset + PPO_BATCH_SIZE], axis=-1)
                batch_ref_v = traj_ref_v[batch_offset:batch_offset + PPO_BATCH_SIZE]
                batch_old_log_prob_v = old_log_prob_v[batch_offset:batch_offset + PPO_BATCH_SIZE]

                with tf.GradientTape() as crt_grad:
                    value_v = net_crt(states_v)
                    loss_value_v = tf.keras.losses.MSE(batch_ref_v, tf.squeeze(value_v, axis=-1))

                gradients = crt_grad.gradient(loss_value_v, net_crt.trainable_variables)
                crt_opt.apply_gradients(zip(gradients, net_crt.trainable_variables))

                with tf.GradientTape() as act_grad:
                    mu_v = net_act(states_v)
                    log_prob_pi_v = calc_log_prob(mu_v, net_act.logstd, actions_v)
                    ratio_v = tf.exp(log_prob_pi_v - batch_old_log_prob_v)
                    surr_obj_v = batch_adv_v * ratio_v
                    clipped_surr_v = batch_adv_v * tf.clip_by_value(ratio_v, 1.0 - PPO_EPS, 1.0 + PPO_EPS)
                    loss_policy_v = -tf.reduce_mean(tf.minimum(surr_obj_v, clipped_surr_v))

                act_gradients = act_grad.gradient(loss_policy_v, net_act.trainable_variables)
                act_opt.apply_gradients(zip(act_gradients, net_act.trainable_variables))

                sum_loss_value += loss_value_v.numpy()
                sum_loss_policy += loss_policy_v.numpy()

                count_steps += 1

        trajectory.clear()




