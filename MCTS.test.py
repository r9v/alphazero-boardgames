from MCTS import MCTS
import unittest
import numpy as np

from Connect4Game import Connect4Game
from TTT import TTTGame

game = TTTGame()


class TestMCTS(unittest.TestCase):

    def testGetPolicy(self):
        mcts = MCTS(game, None)
        board, currentPlayer = game.newGame()
        node = mcts.getPolicy(200, board, currentPlayer)
        print(node)


if __name__ == '__main__':
    unittest.main()
