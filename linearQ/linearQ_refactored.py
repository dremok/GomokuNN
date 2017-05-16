# -*- coding: utf-8 -*-

import random
from collections import namedtuple

import countpat
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def get_features():
    features = [[0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, -1, -1, 0],
                [0, -1, -1, -1, 0],
                [0, -1, -1, -1, -1, 0],
                [0, -1, -1, 0, -1, 0],
                [-1, -1, 0, -1, -1],
                [-1, -1, -1, 0, -1],
                [1, -1, -1, -1, -1, 0]]
    features = [np.array([feature], dtype=np.int) for feature in features]

    w = np.array(
        [0.1360, 0.2662, 0.9232, 0.1967, 1.5843, -0.1594, -0.6747, -1.3054, -1.3054, -0.7254, -0.5805, -1.1453])

    # diagonal versions:
    f_diag = []
    w_diag = []
    for i in range(0, len(features)):
        pattern = features[i]
        if pattern.shape[0] == 1:
            f_ = 2 * np.ones((pattern.shape[1], pattern.shape[1]), dtype=np.int)
            f_ = f_ - np.diag(np.diag(f_)) + np.diag(pattern[0])
            f_diag.append(f_)
            w_diag.append(w[i])
    features = features + f_diag
    w = np.concatenate((w, w_diag))

    f_all = []
    w_all = []
    f_group = []
    for i in range(0, len(features)):
        patterns_for_feature = symmetrical_patterns(features[i])
        patterns_for_feature = remove_redundant_symmetries(patterns_for_feature)
        f_all = f_all + patterns_for_feature
        w_all = w_all + len(patterns_for_feature) * [w[i]]
        f_group = f_group + len(patterns_for_feature) * [i]

    f = {'pattern': f_all, 'group': np.array(f_group, dtype=int)}
    w = np.zeros((max(f['group']) + 1,))
    for k in range(0, len(f['pattern'])):
        w[f['group'][k]] = w_all[k]
    return f, w


def symmetrical_patterns(x):
    return [x, LR(x), UD(x), LR(UD(x)), T(x), LR(T(x)), UD(T(x)), LR(UD(T(x)))]


def T(x):
    y = np.zeros((x.shape[1], x.shape[0]), dtype=int)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            y[j, i] = x[i, j]
    return y


def LR(x):
    y = np.zeros(x.shape, dtype=int)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            y[i, j] = x[i, x.shape[1] - j - 1]
    return y


def UD(x):
    y = np.zeros(x.shape, dtype=int)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            y[i, j] = x[x.shape[0] - i - 1, j]
    return y


def remove_redundant_symmetries(patterns):
    result = []
    for j in range(0, len(patterns)):
        is_redundant = False
        for k in range(0, len(result)):
            if result[k].shape == patterns[j].shape:
                if np.all(result[k] == patterns[j]):
                    is_redundant = True
        if not is_redundant:
            result.append(patterns[j])
    return result


StateTransition = namedtuple('StateTransition', 'state action reward candidates candidate_phis')


def roll_out(w, f, epsilon):
    n = 13
    board = np.zeros((n, n), dtype=int)
    is_over = 0
    player = 1

    game_record = []
    while not is_over:
        if np.all(board):
            # draw
            break

        # state (board, player)
        state = (np.array(board), player)

        # choose action
        chosen_action, candidate_actions, candidate_phis = choose_action(state, w, f, epsilon)

        # update board
        board[chosen_action[0], chosen_action[1]] = player

        # check win:
        is_win = check_win(board, chosen_action)
        if is_win:
            is_over = 1
            reward = 1
        else:
            reward = 0

        # new entry in game record: (s,a,r,a_,phi_a_)
        game_record.append(StateTransition(state, chosen_action, reward, candidate_actions, np.array(candidate_phis)))

        # next player's turn
        player = -player
    return game_record


def check_win(b, a):
    WIN_CONDITION = 5

    def check_win_1d(x):
        score = np.convolve(x, np.ones(WIN_CONDITION, dtype=np.int))
        return max(abs(score)) == WIN_CONDITION

    (i, j) = a
    win = check_win_1d(b[i, :])
    if not win:
        win = check_win_1d(b[:, j])
    if not win:
        win = check_win_1d(b.diagonal(j - i))
    if not win:
        win = check_win_1d(np.fliplr(b).diagonal((b.shape[1] - j - 1) - i))
    return win


def choose_action(s, w, f, epsilon):
    # output:
    # a: the chosen action,
    # a_: list of candidate actions,
    # phi_a_: phi-value for each candidate)
    board = s[0]
    if not board.any():
        # empty board, play in the middle:
        n = board.shape[0]
        jMid = int((n + 1) / 2 - 1)
        a = (jMid, jMid)
        candidate_actions = [a]
        chandidate_phis = np.zeros((1, len(w)), dtype=np.int)
    else:
        a, _, candidate_actions, chandidate_phis = argmaxQ(s, f, w)
        if random.random() < epsilon:
            # choose random action:
            a = candidate_actions[random.randint(0, len(candidate_actions) - 1)]
    return a, candidate_actions, chandidate_phis


def argmaxQ(s, f, w):
    board = s[0]

    # get all possible actions:
    def conv2(x, y, mode='same'):
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

    c = conv2(abs(board), np.ones((3, 3), dtype=np.int))
    candidate_actions = [(a[0], a[1]) for a in np.argwhere((c > 0) & (board == 0))]

    # calculate phi(s,a) for all possible actions:
    action_count = len(candidate_actions)
    phi_a = np.zeros((action_count, len(w)), dtype=np.int)

    for j in range(0, len(candidate_actions)):
        phi_a[j] = phi(s, candidate_actions[j], f)

    # calculate Q(s,a) for all possible actions:
    Q = phi_a @ w

    # find the best action (ties resolved randomly)
    m = np.max(Q)
    ii = np.where(Q > m - 1e-6)[0]
    if len(ii) > 1:
        k = ii[random.sample(range(0, len(ii)), 1)[0]]
    else:
        k = ii[0]
    a_opt = candidate_actions[k]
    Q_opt = Q[k]
    return a_opt, Q_opt, candidate_actions, phi_a


def phi(s, a, f):
    board = s[0]
    player = s[1]
    board[a[0], a[1]] = player
    count = count_pattern(board, player, f)
    board[a[0], a[1]] = 0
    return count


def count_pattern(b, p, f):
    # b: board
    # p: player
    # f: features
    b_ = b * p
    c_ = np.array([0], dtype=int)
    c = np.zeros(np.max(f['group']) + 1, dtype=np.int)
    for j in range(0, len(f['pattern'])):
        countpat.countpat_func(c_, b_, f['pattern'][j])
        c[f['group'][j]] += c_[0]
    return c


class Adam:
    a = 0.001
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    n = 0

    def __init__(self, w):
        self.m = np.zeros_like(w)
        self.v = np.zeros_like(w)

    def update(self, w, dLdw):
        self.n = self.n + 1
        self.m = self.b1 * self.m + (1 - self.b1) * dLdw
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(dLdw, 2)
        mHat = self.m / (1 - self.b1 ** self.n)
        vHat = self.v / (1 - self.b2 ** self.n)
        return w + self.a * mHat / (np.sqrt(vHat) + self.eps)


ExperienceItem = namedtuple('ExperienceItem', 'state action reward next_state next_candidates next_candidate_phis')


class Experience:
    write_idx = 0  # where to write next item
    item_count = 0  # total number of items currently in memory
    max_items = 10000
    data = []

    def add(self, x):
        if self.write_idx < self.max_items:
            # self.data[self.n] = x
            self.data.append(x)
            self.write_idx = self.write_idx + 1
        else:
            self.data[0] = x
            self.write_idx = 1
        self.item_count = max(self.write_idx, self.item_count)

    def store_game(self, game_record):
        # all but last state transition for first player:
        for j in range(0, len(game_record) - 2, 2):
            player1_transition = game_record[j]
            player1_next_transition = game_record[j + 2]
            item = ExperienceItem(player1_transition.state,  # s
                                  player1_transition.action,  # a
                                  player1_transition.reward,  # r
                                  player1_next_transition.state,  # s_next
                                  player1_next_transition.candidates,  # a_next (candidate moves at s_next)
                                  player1_next_transition.candidate_phis)  # phi_next (phi-values at s_next)
            self.add(item)
        # all but last state transition for second player:
        for j in range(1, len(game_record) - 2, 2):
            player2_transition = game_record[j]
            player2_next_transition = game_record[j + 2]
            item = ExperienceItem(player2_transition.state,  # s
                                  player2_transition.action,  # a
                                  player2_transition.reward,  # r
                                  player2_next_transition.state,  # s_next
                                  player2_next_transition.candidates,  # a_next (candidate moves at s_next)
                                  player2_next_transition.candidate_phis)  # phi_next (phi-values at s_next)
            self.add(item)
        # last state transition for loser:
        self.add(ExperienceItem(game_record[-2].state,  # s
                                game_record[-2].action,  # a
                                -game_record[-1].reward,  # r (-1 times winner's reward)
                                None, None, None))  # "None" here because final state
        # last state transition for winner:
        self.add(ExperienceItem(game_record[-1].state,  # s
                                game_record[-1].action,  # a
                                game_record[-1].reward,  # r
                                None, None, None))  # "None" here because final state


def main():
    mini_batch_size = 50
    epsilon = 0.1

    # features, initial weights:
    features, w = get_features()

    # adam:
    adam = Adam(w)

    # initialise experience:
    experience = Experience()
    while experience.item_count < mini_batch_size:
        game_record = roll_out(w, features, epsilon)
        experience.store_game(game_record)

    wAll = []
    LAll = []

    # start learning:
    for i in range(0, 3000):
        # play a game:
        game_record = roll_out(w, features, epsilon)

        # add it to experience:
        experience.store_game(game_record)

        # stochastic gradient descent
        for _ in range(0, 5):
            # mini-batch:
            L = 0  # loss
            dLdw = np.zeros_like(w)  # gradient
            for k in random.sample(range(0, experience.item_count), mini_batch_size):
                state, action, reward, next_state, next_candidates, next_candidate_phis = experience.data[k]
                if next_state is None:
                    # final state:
                    y = reward
                else:
                    Q = next_candidate_phis @ w
                    y = np.max(Q)
                phi_a = phi(state, action, features)
                x = (y - np.dot(w, phi_a))
                L = L + x ** 2
                dLdw = dLdw + x * phi_a
            L = -L / mini_batch_size
            dLdw = dLdw / mini_batch_size

            # adam step:
            w = adam.update(w, dLdw)

            wAll.append(np.array(w))
            LAll.append(L)
        if i % 10 == 0:
            print(i)

    w_ = np.empty((len(wAll), len(w)))
    L_ = np.empty((len(LAll),))

    for j in range(0, len(wAll)):
        w_[j][:] = wAll[j]
        L_[j] = LAll[j]

    figure = plt.figure(1)
    plt.clf()
    plt.subplot(121)
    plt.plot(w_)
    plt.subplot(122)
    plt.plot(L_)
    figure.savefig('foo.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
