import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from game.board import Board


class RandomAgent:
    def __init__(self):
        pass

    def action(self, board_state: Board):
        random_choice = random.choice(board_state.legal_moves())
        x = random_choice[0]
        y = random_choice[1]
        return x, y


def test_agent():
    agent = RandomAgent()

    for i in range(20):
        board = Board(n=5)
        outcome = 0
        player = 0
        while not outcome:
            x, y = agent.action(board)
            state, outcome = board.update(player, x, y)
            player ^= 1

        print(board)
        if outcome == 1:
            print('Player wins!')
        else:
            print('It\'s a tie...')
        print()


if __name__ == '__main__':
    test_agent()