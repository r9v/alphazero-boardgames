from TTTNet.NNetWrapper import NNetWrapper as NNet
from MCTS import MCTS
import unittest
import numpy as np

from Connect4Game import Connect4Game
from TTT import TTTGame

game = TTTGame()
net = NNet()
net.load_checkpoint()


class TestMCTS(unittest.TestCase):

    def testGetPolicy(self):
        mcts = MCTS(game, net)
        state = game.newGame()
        state.board[0][0] = -1
        state.board[0][1] = 0
        state.board[0][2] = 0

        state.board[1][0] = 0
        state.board[1][1] = 0
        state.board[1][2] = 0

        state.board[2][0] = 0
        state.board[2][1] = 0
        state.board[2][2] = 1

        state.player = -1
        state.availableActions = state._availableActions()
        node = mcts.getPolicy(230, state)
        print(node)


if __name__ == '__main__':
    unittest.main()
