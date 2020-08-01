from TTT import TTTGame
import unittest
import numpy as np


class TestTTTGame(unittest.TestCase):

    def testNewGame(self):
        game = TTTGame()
        expected_board = np.zeros((3, 3), dtype="int")
        board, player = game.newGame()
        self.assertTrue(np.array_equal(board, expected_board))
        self.assertEqual(player, -1)

    def testStep(self):
        game = TTTGame()
        board, player = game.newGame()
        board, player = game.step(board, player, 0)
        board, player = game.step(board, player, 4)
        board, player = game.step(board, player, 8)

        expected_board = np.zeros((3, 3), dtype="int")
        expected_board[0][0] = -1
        expected_board[1][1] = 1
        expected_board[2][2] = -1

        self.assertTrue(np.array_equal(board, expected_board))
        self.assertEqual(player, 1)

    def testOver(self):
        game = TTTGame()
        board, player = game.newGame()
        over, winer = game.over(board)
        self.assertFalse(over)
        self.assertIsNone(winer)

        board = np.zeros((3, 3), dtype="int")
        board[0][0] = 1
        board[1][1] = 1
        board[2][2] = 1

        over, winer = game.over(board)
        self.assertTrue(over)
        self.assertEqual(winer, 1)

        board[0][0] = -1
        board[1][1] = -1
        board[2][2] = -1

        over, winer = game.over(board)
        self.assertTrue(over)
        self.assertEqual(winer, -1)


if __name__ == '__main__':
    unittest.main()
