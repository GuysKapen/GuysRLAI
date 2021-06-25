import os
import os
import time

import pybullet_envs
import gym
import numpy as np
from tensorflow_dl.libs import experience
from tensorflow_dl.notes_book.actor_critic.continous_action_space.agent import AgentDDPG
from tensorflow_dl.notes_book.actor_critic.continous_action_space.model import *

ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100
REPLAY_INITIAL = 10
ALPHA = 1 - 1e-3
TEST_ITERS = 1000

Vmax = 10
Vmin = -10
N_ATOMS = 128
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


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


def distr_projection(next_distr_v, rewards_v, dones_mask, gamma):
    next_distr = next_distr_v.numpy()
    rewards = rewards_v.numpy()
    dones_mask = dones_mask.numpy()
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)

    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
        b_j = (tz_j - Vmin) / DELTA_Z
        lower = np.floor(b_j).astype(np.int64)
        upper = np.ceil(b_j).astype(np.int64)
        eq_mask = upper == lower
        proj_distr[eq_mask, lower[eq_mask]] += next_distr[eq_mask, atom]
        # Take value of lower and upper with mask and add to proj_distr (boolean index - true is take the value in
        # this index (the boolean index), false otherwise
        ne_mask = upper != lower
        proj_distr[ne_mask, lower[ne_mask]] += next_distr[ne_mask, atom] * (upper - b_j)[ne_mask]
        proj_distr[ne_mask, upper[ne_mask]] += next_distr[ne_mask, atom] * (b_j - lower)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
        b_j = (tz_j - Vmin) / DELTA_Z
        lower = np.floor(b_j).astype(np.int64)
        upper = np.ceil(b_j).astype(np.int64)
        eq_mask = upper == lower
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, lower[eq_mask]] = 1.0
        ne_mask = upper != lower
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, lower[ne_mask]] = (upper - b_j)[ne_mask]
            proj_distr[ne_dones, upper[ne_mask]] = (b_j - lower)[ne_mask]
    return tf.convert_to_tensor(proj_distr)


def unpack_batch_ddqn(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = tf.convert_to_tensor(states, dtype=tf.float32)
    actions_v = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards_v = tf.convert_to_tensor(rewards, dtype=tf.float32)
    last_states_v = tf.convert_to_tensor(last_states, dtype=tf.float32)
    dones_t = tf.convert_to_tensor(dones, dtype=tf.bool)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


def arrays_multi_alpha_blend(arr1, arr2, alpha):
    return [x * alpha + arr2[i] * (1 - alpha) for i, x in enumerate(arr1)]
    # return list(map(lambda x: x * alpha, arr1))


if __name__ == '__main__':
    spec = gym.envs.registry.spec(ENV_ID)
    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)

    save_path = "/content/data/MyDrive/models/RobotD4PG"
    save_path_critic = "/content/data/MyDrive/models/RobotDDPGCritic"
    if os.path.exists(save_path):
        act_net = tf.keras.models.load_model(save_path)
        print("#" * 60)
        print("Restored actor saved model!")
    else:
        act_net = DDPGActor(env.action_space.shape[0])
    if os.path.exists(save_path_critic):
        crt_net = tf.keras.models.load_model(save_path)
        print("#" * 60)
        print("Restored critic saved model!")
    else:
        crt_net = D4PGCritic(env.action_space.shape[0], N_ATOMS, Vmin, Vmax)

    target_act_net = DDPGActor(env.action_space.shape[0])
    target_crt_net = D4PGCritic(env.action_space.shape[0], N_ATOMS, Vmin, Vmax)

    agent = AgentDDPG(act_net)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    writer = tf.summary.create_file_writer("d4pg-robot")

    best_reward = None
    batch = []
    step_idx = 0
    while True:
        step_idx += 1
        buffer.populate(1)
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            rewards, steps = zip(*rewards_steps)

        if len(buffer) < REPLAY_INITIAL:
            continue

        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, rewards_v, dones_mask, last_states_v = unpack_batch_ddqn(batch)

        with tf.GradientTape(persistent=True) as g:
            crt_distr_v = crt_net((states_v, actions_v))
            last_act_v = target_act_net(last_states_v)
            last_distr_v = tf.math.softmax(target_crt_net((last_states_v, last_act_v)), axis=1)
            proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask, gamma=GAMMA**2)
            proj_distr_v = -tf.math.log_softmax(crt_distr_v, axis=1) * proj_distr_v
            critic_loss_v = tf.reduce_mean(tf.reduce_sum(proj_distr_v, axis=1))

            # actor
            cur_actions_v = act_net(states_v)
            crt_distr_v = crt_net((states_v, cur_actions_v))
            actor_loss_v = -crt_net.distr_to_q(crt_distr_v)
            actor_loss_v = tf.reduce_mean(actor_loss_v)

        critic_gradients = g.gradient(critic_loss_v, crt_net.trainable_variables)
        optimizer.apply_gradients(zip(critic_gradients, crt_net.trainable_variables))

        actor_gradients = g.gradient(actor_loss_v, act_net.trainable_variables)
        optimizer.apply_gradients(zip(actor_gradients, act_net.trainable_variables))

        target_act_net.set_weights(arrays_multi_alpha_blend(target_act_net.get_weights(), act_net.get_weights(), ALPHA))
        target_crt_net.set_weights(arrays_multi_alpha_blend(target_crt_net.get_weights(), crt_net.get_weights(), ALPHA))

        if step_idx % TEST_ITERS == 0:
            ts = time.time()
            rewards, steps = test_net(act_net, test_env)
            print("Test done in %.2f sec, reward %.3f, steps %d" % (
                time.time() - ts, rewards, steps))
            with writer.as_default():
                tf.summary.scalar("test_reward", rewards, step_idx)
                tf.summary.scalar("test_steps", steps, step_idx)
            if best_reward is None or best_reward < rewards:
                if best_reward is not None:
                    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                    name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                    fname = os.path.join(save_path, name)
                    act_net.save(save_path)
                    crt_net.save(save_path_critic)
                best_reward = rewards
