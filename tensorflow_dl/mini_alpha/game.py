"""
4-in-a-row game-related functions.
Field is 6*7 with pieces falling from the top to the bottom. There are two kinds of pieces: black and white,
which are encoded by 1 (black) and 0 (white).
There are two representation of the game:
1. List of 7 lists with elements ordered from the bottom. For example, this field
0     1
0     1
10    1
10  0 1
10  1 1
101 111
Will be encoded as [
  [1, 1, 1, 1, 0, 0],
  [0, 0, 0, 0],
  [1],
  [],
  [1, 1, 0],
  [1],
  [1, 1, 1, 1, 1, 1]
]
2. integer number consists from:
    a. 7*6 bits (column-wise) encoding the field. Unoccupied bits are zero
    b. 7*3 bits, each 3-bit number encodes amount of free entries on the top.
In this representation, the field above will be equal to those bits:
[
    111100,
    000000,
    100000,
    000000,
    110000,
    100000,
    111111,
    000,
    010,
    101,
    110,
    011,
    101,
    000
]
All the code is generic, so, in theory you can try to adjust the field size.
But tests could become broken.
"""
import collections

import numpy as np
import tensorflow as tf

from tensorflow_dl.mini_alpha import mcts
from tensorflow_dl.mini_alpha.model import Net

GAME_ROWS = 6
GAME_COLS = 7
BITS_IN_LEN = 3
PLAYER_BLACK = 1
PLAYER_WHITE = 0
COUNT_TO_WIN = 4
OBS_SHAPE = (2, GAME_ROWS, GAME_COLS)


# declared after encode_lists


def bits_to_int(bits):
    res = 0
    for b in bits:
        res *= 2
        res += b
    return res


def int_to_bits(num, bits):
    res = []
    for _ in range(bits):
        res.append(num % 2)
        num //= 2
    return res[::-1]


# def int_to_bits(num, bits):
#     return f'{num:0{bits}b}'


def encode_lists(field_lists):
    """
    Encode lists representation into the binary numbers
    :param field_lists: list of GAME_COLS lists with 0s and 1s
    :return: integer number with encoded game state
    """
    assert isinstance(field_lists, list)
    assert len(field_lists) == GAME_COLS

    bits = []
    len_bits = []
    for col in field_lists:
        bits.extend(col)
        free_len = GAME_ROWS - len(col)
        bits.extend([0] * free_len)
        len_bits.extend(int_to_bits(free_len, bits=BITS_IN_LEN))
    bits.extend(len_bits)
    return bits_to_int(bits)


INITIAL_STATE = encode_lists([[]] * GAME_COLS)


def decode_binary(state_int):
    """
    Decode binary representation into the list view
    :param state_int: integer representing the field
    :return: list of GAME_COLS lists
    """
    assert isinstance(state_int, int)
    bits = int_to_bits(state_int, bits=GAME_COLS * GAME_ROWS + GAME_COLS * BITS_IN_LEN)
    res = []
    len_bits = bits[GAME_COLS * GAME_ROWS:]
    for col in range(GAME_COLS):
        vals = bits[col * GAME_ROWS:(col + 1) * GAME_ROWS]
        lens = bits_to_int(len_bits[col * BITS_IN_LEN:(col + 1) * BITS_IN_LEN])
        if lens > 0:
            vals = vals[:-lens]
        res.append(vals)
    return res


def possible_moves(state_int):
    """
    This function could be calculated directly from bits, but I'm too lazy
    :param state_int: field representation
    :return: the list of columns which we can make a move
    """
    assert isinstance(state_int, int)
    field = decode_binary(state_int)
    return [idx for idx, col in enumerate(field) if len(col) < GAME_ROWS]


def _check_won(field, col, delta_row):
    """
    Check for horisontal/diagonal win condition for the last player moved in the column
    :param field: list of lists
    :param col: column index
    :param delta_row: if 0, checks for horisonal won, 1 for rising diagonal, -1 for falling
    :return: True if won, False if not
    """
    player = field[col][-1]
    coord = len(field[col]) - 1
    total = 1
    # negative dir
    cur_coord = coord - delta_row
    for c in range(col - 1, -1, -1):
        if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= GAME_ROWS:
            break
        if field[c][cur_coord] != player:
            break
        total += 1
        if total == COUNT_TO_WIN:
            return True
        cur_coord -= delta_row
    # positive dir
    cur_coord = coord + delta_row
    for c in range(col + 1, GAME_COLS):
        if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= GAME_ROWS:
            break
        if field[c][cur_coord] != player:
            break
        total += 1
        if total == COUNT_TO_WIN:
            return True
        cur_coord += delta_row
    return False


def move(state_int, col, player):
    """
    Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
    :param state_int: current state
    :param col: column to make a move
    :param player: player index (PLAYER_WHITE or PLAYER_BLACK
    :return: tuple of (state_new, won). Value won is bool, True if this move lead
    to victory or False otherwise (but it could be a draw)
    """
    assert isinstance(state_int, int)
    assert 0 <= col < GAME_COLS
    assert player == PLAYER_BLACK or player == PLAYER_WHITE
    field = decode_binary(state_int)
    assert len(field[col]) < GAME_ROWS
    field[col].append(player)
    # check for victory: the simplest vertical case
    suff = field[col][-COUNT_TO_WIN:]
    won = suff == [player] * COUNT_TO_WIN
    if not won:
        won = _check_won(field, col, 0) or _check_won(field, col, 1) or _check_won(field, col, -1)
    state_new = encode_lists(field)
    return state_new, won


def render(state_int):
    state_list = decode_binary(state_int)
    data = [[' '] * GAME_COLS for _ in range(GAME_ROWS)]
    for col_idx, col in enumerate(state_list):
        for rev_row_idx, cell in enumerate(col):
            row_idx = GAME_ROWS - rev_row_idx - 1
            data[row_idx][col_idx] = str(cell)
    return [''.join(row) for row in data]


def update_counts(counts_dict, key, counts):
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res


def state_lists_to_batch(state_lists, who_moves_lists):
    """
    Convert list of list states to batch for network
    :param state_lists: list of 'list states'
    :param who_moves_lists: list of player index who moves
    :return Variable with observations
    """
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + OBS_SHAPE, dtype=np.float32)
    for idx, (state, who_move) in enumerate(zip(state_lists, who_moves_lists)):
        _encode_list_state(batch[idx], state, who_move)
    return tf.convert_to_tensor(batch)


def _encode_list_state(dest_np, state_list, who_move):
    """
    In-place encodes list state into the zero numpy array
    :param dest_np: dest array, expected to be zero
    :param state_list: state of the game in the list form
    :param who_move: player index (game.PLAYER_WHITE or game.PLAYER_BLACK) who to move
    """
    assert dest_np.shape == OBS_SHAPE

    for col_idx, col in enumerate(state_list):
        for rev_row_idx, cell in enumerate(col):
            row_idx = GAME_ROWS - rev_row_idx - 1
            if cell == who_move:
                dest_np[0, row_idx, col_idx] = 1.0
            else:
                dest_np[1, row_idx, col_idx] = 1.0


def play_game(mcts_stores, replay_buffer, net1, net2, steps_before_tau_0, mcts_searches, mcts_batch_size,
              net1_plays_first=None):
    """
        Play one single game, memorizing transitions into the replay buffer
        :param mcts_batch_size: mcts batch size for running search
        :param mcts_searches: mcts num search
        :param steps_before_tau_0:
        :param net1_plays_first: whether 1 play first
        :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
        :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
        :param net1: player1
        :param net2: player2
        :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """
    assert isinstance(replay_buffer, (collections.deque, type(None)))
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = INITIAL_STATE
    nets = [net1, net2]

    if net1_plays_first is None:
        cur_player = np.random.choice(2)
    else:
        cur_player = 0 if net1_plays_first else 1
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0

    game_history = []

    result = None
    net1_result = None

    while result is None:
        mcts_stores[cur_player].search_batch(mcts_searches, mcts_batch_size, state, cur_player, nets[cur_player])

        probs, _ = mcts_stores[cur_player].get_policy_value(state, tau)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(GAME_COLS, p=probs)
        if action not in possible_moves(state):
            print("Select impossible action")

        state, won = move(state, action, cur_player)
        if won:
            result = 1
            net1_result = 1 if cur_player == 0 else -1
            break
        cur_player = 1 - cur_player

        # check draw
        if len(possible_moves(state)) == 0:
            result = 0
            net1_result = 0
            break

        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append((state, cur_player, probs, result))
            result = -result

    return net1_result, step
