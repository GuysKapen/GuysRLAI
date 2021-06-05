import os
import gym
import numpy as np
import argparse

import tensorflow as tf

from tensorflow_dl.notes_book.trading import environment, data, models
from tensorflow_dl.libs import actions, agent, experience, common

BATCH_SIZE = 32
BARS_COUNT = 10
TARGET_NET_SYNC = 1000
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"
DEFAULT_VAL_STOCKS = "data/YNDX_150101_151231.csv"

GAMMA = 0.99

REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000

REWARD_STEPS = 2

LEARNING_RATE = 0.0001

STATES_TO_EVALUATE = 1000
EVAL_EVERY_STEP = 1000

EPSILON_START = 1.0
EPSILON_STOP = 0.1
EPSILON_STEPS = 1000000

CHECKPOINT_EVERY_STEP = 1000000
VALIDATION_EVERY_STEP = 100000

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), "data", "YNDX_160101_161231.csv")
    val_path = os.path.join(os.getcwd(), "data", "YNDX_150101_151231.csv")
    saves_path = ""
    stock_data = {"YNDX": data.load_relative(data_path)}
    env = environment.TradingEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)
    env_tst = environment.TradingEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    val_data = {"YNDX": data.load_relative(val_path)}
    env_val = environment.TradingEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)

    summary = tf.summary.create_file_writer("summary")
    net = models.SimpleFFDQN()
    tgt_net = models.SimpleFFDQN()

    selector = actions.EpsilonGreedyActionSelector(EPSILON_START)
    train_agent = agent.DQNAgent(net, selector)
    exp_source = experience.ExperienceSourceFirstLast(env, train_agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    step_idx = 0
    eval_states = None
    best_mean_val = None

    with common.RewardTracker(summary, np.inf, group_rewards=100) as reward_tracker:
        while True:
            step_idx += 1
            buffer.populate(1)
            selector.epsilon = max(EPSILON_STOP, EPSILON_START - step_idx / EPSILON_STEPS)

            new_rewards = exp_source.pop_rewards_steps()
            if new_rewards:
                reward_tracker.reward(new_rewards[0], step_idx, selector.epsilon)

            if len(buffer) < REPLAY_INITIAL:
                continue

            if eval_states is None:
                print("Initial buffer populated, start training")
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            if step_idx % EVAL_EVERY_STEP == 0:
                mean_val = common.calc_values_of_states(eval_states, net)
                with summary.as_default():
                    tf.summary.scalar("values_mean", mean_val, step_idx)
                if best_mean_val is None or best_mean_val < mean_val:
                    if best_mean_val is not None:
                        print("%d: Best mean value updated %.3f -> %.3f" % (step_idx, best_mean_val, mean_val))
                    best_mean_val = mean_val
                    net.save(saves_path)

            batch = buffer.sample(BATCH_SIZE)
            with tf.GradientTape() as g:
                loss_v = common.calc_loss(batch, net, tgt_net, GAMMA ** REWARD_STEPS)

            gradients = g.gradient(loss_v, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))

            if step_idx % TARGET_NET_SYNC == 0:
                tgt_net.set_weights(net.get_weights())

            if step_idx % CHECKPOINT_EVERY_STEP == 0:
                idx = step_idx // CHECKPOINT_EVERY_STEP
                net.save(saves_path)

            # if step_idx % VALIDATION_EVERY_STEP == 0:
            #     res =