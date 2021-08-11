import collections
import random
import time

import tensorflow as tf

from tensorflow_dl.mini_alpha import game, model, mcts

PLAY_EPISODES = 1  # 25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER = 5000  # 30000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000  # 10000

BEST_NET_WIN_RATIO = 0.60

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 10


def evaluate(net1, net2, rounds):
    n1_win = n2_win = 0
    mcts_stores = [mcts.MCTS(), mcts.MCTS()]

    for idx in range(rounds):
        r, _ = game.play_game(mcts_stores, replay_buffer=None, net1=net1, net2=net2, steps_before_tau_0=0,
                              mcts_searches=20, mcts_batch_size=16)

        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1

    return n1_win / (n1_win + n2_win)


if __name__ == '__main__':
    net = model.Net(actions_n=game.GAME_COLS)
    best_net = model.Net(actions_n=game.GAME_COLS)

    optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    mcts_store = mcts.MCTS()
    step_idx = 0
    best_idx = 0

    while True:
        t = time.time()
        prev_nodes = len(mcts_store)
        game_steps = 0
        for _ in range(PLAY_EPISODES):
            _, steps = game.play_game(mcts_store, replay_buffer, best_net, best_net,
                                      steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                                      mcts_batch_size=MCTS_BATCH_SIZE)
            game_steps += steps

        game_nodes = len(mcts_store) - prev_nodes
        dt = time.time() - t
        speed_steps = game_steps / dt
        speed_nodes = game_nodes / dt
        print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d, replay %d" % (
            step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx, len(replay_buffer)))
        step_idx += 1

        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue

        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0

        for _ in range(TRAIN_ROUNDS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
            batch_states_lists = [game.decode_binary(state) for state in batch_states]
            states_v = game.state_lists_to_batch(batch_states_lists, batch_who_moves)

            probs_v = tf.convert_to_tensor(batch_probs, dtype=tf.float32)
            values_v = tf.convert_to_tensor(batch_values, dtype=tf.float32)

            with tf.GradientTape() as g:

                out_logits_v, out_values_v = net(states_v)

                loss_value_v = tf.keras.losses.MSE(values_v, tf.squeeze(out_values_v, axis=-1))
                loss_policy_v = -tf.math.log_softmax(out_logits_v, axis=1) * probs_v
                loss_policy_v = tf.reduce_mean(tf.reduce_sum(loss_policy_v, axis=1))

                loss_v = loss_policy_v + loss_value_v

            gradients = g.gradient(loss_v, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))

            sum_loss += tf.reduce_mean(loss_v)
            sum_value_loss += loss_value_v
            sum_policy_loss += loss_policy_v

        if step_idx % EVALUATE_EVERY_STEP == 0:
            win_ratio = evaluate(net, best_net, rounds=EVALUATION_ROUNDS)
            print("Net evaluated, win ratio = %.2f" % win_ratio)
            if win_ratio > BEST_NET_WIN_RATIO:
                print("Net is better than cur best, sync")

                best_idx += 1
                best_net.set_weights(net.get_weights())
                mcts_store.clear()



