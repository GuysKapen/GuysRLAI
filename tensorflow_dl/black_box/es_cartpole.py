import gym
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
import time

MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 10000
NOISE_STD = 0.01
LEARNING_RATE = 0.001


class Net(tf.keras.Model):
    def get_config(self):
        return {'action_size': self.action_size}

    def __init__(self, action_size):
        super(Net, self).__init__()
        self.action_size = action_size
        self.seq = tf.keras.Sequential([
            layers.Dense(units=32),
            layers.ReLU(),
            layers.Dense(action_size),
            layers.Softmax(axis=1)
        ])

    def call(self, inputs, training=None, mask=None):
        return self.seq(inputs)


def evaluate(env, net):
    obs = env.reset()
    reward = 0.0
    steps = 0
    while True:
        obs_v = tf.convert_to_tensor([obs])
        acts = net(obs_v)
        acts = tf.argmax(acts, axis=1)
        obs, r, done, _ = env.step(acts.numpy()[0])
        reward += r
        steps += 1
        if done:
            break
    return reward, steps


def sample_noise(net):
    pos = []
    neg = []
    for p in net.trainable_variables:
        noise_t = tf.random.normal(shape=p.shape, dtype=tf.float32)
        pos.append(noise_t)
        neg.append(-noise_t)
    return pos, neg


def evaluate_with_noise(env, net, noises):
    scaled_noises = []
    for param, noise in zip(net.trainable_variables, noises):
        scaled_noise = NOISE_STD * noise
        param.assign_add(scaled_noise)
        scaled_noises.append(scaled_noise)
    reward, steps = evaluate(env, net)
    for param, scaled_noise in zip(net.trainable_variables, scaled_noises):
        param.assign_sub(scaled_noise)
    return reward, steps


def train_step(net, batch_noise, batch_reward):
    """
    Train the net work with objective 0_t+1 = 0_t + a * 1/(n*o) * sum_i=1->n(F_i*e_i)
    with: 0_t is net parameter, a - alpha or lr, o: std, F_i: fitness value (or reward), e_i: noise
    :param net:
    :param batch_noise:
    :param batch_reward:
    :param step_idx:
    :return:
    """
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    std = np.std(norm_reward)
    if abs(std) > 1e-6:
        norm_reward /= std

    weighted_noise = None
    for reward, noise in zip(norm_reward, batch_noise):
        if weighted_noise is None:
            weighted_noise = [reward * p_noise for p_noise in noise]
        else:
            for w_noise, p_noise in zip(weighted_noise, noise):
                w_noise += reward * p_noise

    weights_updated = []
    for param, param_updated in zip(net.trainable_variables, weighted_noise):
        update = param_updated / (len(batch_reward) * NOISE_STD)
        param.assign_add(LEARNING_RATE * update)
        weights_updated.append(param)
    # net.set_weights(weights_updated)


if __name__ == '__main__':
    env = gym.make("CartPole-v0")

    net = Net(env.action_space.n)
    net.build(input_shape=tf.convert_to_tensor([env.reset()]).shape)

    step_idx = 0
    while True:
        t_start = time.time()
        batch_noise = []
        batch_reward = []
        batch_steps = 0
        for _ in range(MAX_BATCH_EPISODES):
            pos_noise, neg_noise = sample_noise(net)
            batch_noise.append(pos_noise)
            batch_noise.append(neg_noise)
            reward, steps = evaluate_with_noise(env, net, pos_noise)
            batch_reward.append(reward)
            batch_steps += steps
            reward, steps = evaluate_with_noise(env, net, neg_noise)
            batch_reward.append(reward)
            batch_steps += steps
            if batch_steps > MAX_BATCH_STEPS:
                break
        
        step_idx += 1
        mean_reward = np.mean(batch_reward)
        if mean_reward > 199:
            print(f"Solved in {step_idx} steps!")
            break

        train_step(net, batch_noise, batch_reward)
        speed = batch_steps / (time.time() - t_start)
        print("%d: reward=%.2f, speed=%.2f f/s" % (step_idx, mean_reward, speed))

