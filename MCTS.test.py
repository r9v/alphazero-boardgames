from MCST import MCTS
import unittest
import numpy as np


class TestMCTS(unittest.TestCase):

    def testGetPolicy(self):
        mcts = MCTS()
        board, currentPlayer = game.newGame()
        mcts.getPolicy(4, board, currentPlayer)


if __name__ == '__main__':
    unittest.main()
