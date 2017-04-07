# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:50:39 2017

@author: Siven
"""

import numpy as np
import re

WIN_CONDITION = 5


def check_win(x):
    score = np.convolve(x, np.ones(WIN_CONDITION, dtype=np.int8))
    win = max(abs(score)) == WIN_CONDITION
    # TODO: Return different values for player 1 win and player 2 win
    return win


class Board:
    def __init__(self, n=13, initial_state=None):
        self.n = n
        self.state = np.zeros((n, n), dtype=np.int8)
        if initial_state:
            self.state[:] = initial_state

    # %% board logic

    def update(self, player, i, j):
        assert self.state[i, j] == 0
        # make move
        if player == 1:
            self.state[i, j] = 1
        else:
            self.state[i, j] = -1

        # check win
        win = check_win(self.state[i, :])
        if not win: win = check_win(self.state[:, j])
        if not win: win = check_win(self.state.diagonal(j - i))
        if not win: win = check_win(np.fliplr(self.state).diagonal((self.n - j - 1) - i))

        # TODO: Return different values for player 1 win, player 2 win, tie and not finished
        outcome = 0
        if win:
            outcome = 1
        elif not self.legal_move_exist():
            outcome = -1
        return self.state, outcome

    def legal_move_exist(self):
        return 0 in self.state

    def legal_moves(self):
        zero_states = np.where(self.state == 0)
        return list(zip(zero_states[0], zero_states[1]))

    def __str__(self):
        board_representation = ' ' + re.sub(r'[\[\]]', '', str(self.state))
        return board_representation


def test_board():
    # %% test it
    board = Board(initial_state=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 1, 0, 1,-1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # no win:
    _, win = board.update(2, 5, 3)
    print(win)
    assert not win

    # row win:
    _, win = board.update(1, 5, 7)
    print(win)
    assert win

    board = Board(initial_state=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1, 1,-1, 1,-1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0,-1, 1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # counter diagonal win:
    _, win = board.update(2, 2, 10)
    print(win)
    assert win

    board = Board(initial_state=[[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # diagonal win:
    _, win = board.update(1, 4, 10)
    print(win)
    assert win


if __name__ == '__main__':
    test_board()
