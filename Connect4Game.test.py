from Connect4Game import Connect4Game
import unittest
import numpy as np


class TestConnect4Game(unittest.TestCase):

    def testPlayFirstColumn(self):
        game = Connect4Game()
        expected_board = np.zeros((6, 7), dtype="int")
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), 0)
        self.assertEqual(game.currentPlayer, -1)

        self.assertFalse(game.step(0)[0])
        expected_board[0][0] = -1
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), 1)
        self.assertEqual(game.currentPlayer, 1)

        expected_board[1][0] = 1
        self.assertFalse(game.step(0)[0])
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), 2)
        self.assertEqual(game.currentPlayer, -1)

        expected_board[2][0] = -1
        self.assertFalse(game.step(0)[0])
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), 3)
        self.assertEqual(game.currentPlayer, 1)

        expected_board[3][0] = 1
        self.assertFalse(game.step(0)[0])
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), 4)
        self.assertEqual(game.currentPlayer, -1)

        expected_board[4][0] = -1
        self.assertFalse(game.step(0)[0])
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), 5)
        self.assertEqual(game.currentPlayer, 1)

        expected_board[5][0] = 1
        self.assertFalse(game.step(0)[0])
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.getFirstFreeRow(0), None)
        self.assertEqual(game.currentPlayer, -1)

        self.assertRaises(Exception, game.step, 0)

    def testStepReset(self):
        game = Connect4Game()
        self.assertRaises(Exception, game.step, -1)
        self.assertRaises(Exception, game.step, 7)

        self.assertFalse(game.step(0)[0])
        self.assertFalse(game.step(6)[0])
        self.assertFalse(game.step(6)[0])

        expected_board = np.zeros((6, 7), dtype="int")
        expected_board[0][0] = -1
        expected_board[0][6] = 1
        expected_board[1][6] = -1

        self.assertEqual(game.currentPlayer, 1)
        self.assertTrue(np.array_equal(game.board, expected_board))

        game.reset()

        expected_board = np.zeros((6, 7), dtype="int")
        self.assertEqual(game.currentPlayer, -1)
        self.assertTrue(np.array_equal(game.board, expected_board))

    def testGameDone(self):
        game = Connect4Game()
        self.assertFalse(game.step(0)[0])
        self.assertFalse(game.step(1)[0])
        self.assertFalse(game.step(0)[0])
        self.assertFalse(game.step(1)[0])
        self.assertFalse(game.step(0)[0])
        self.assertFalse(game.step(1)[0])
        self.assertTrue(game.step(0)[0])
        self.assertTrue(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, -1)

        game = Connect4Game()
        game.board = np.ones((6, 7), dtype="int")
        self.assertTrue(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, None)

        game.board = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype="int")
        self.assertFalse(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, None)

        game.board = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype="int")
        self.assertTrue(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, 1)

        game.board = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, -1, 0],
             [0, 0, 1, 0, -1, 0, 0],
             [0, 0, 0, -1, 0, 0, 0],
             [0, 0, -1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype="int")
        self.assertTrue(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, -1)

        game.board = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0, 0]], dtype="int")
        self.assertTrue(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, 1)

        game.board = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, -1, -1, -1, -1]], dtype="int")
        self.assertTrue(game.checkGameEnded().done)
        self.assertEqual(game.checkGameEnded().victor, -1)

    def testAvailableMoves(self):
        game = Connect4Game()
        game.availableMoves()
        self.assertTrue(np.array_equal(
            game.availableMoves(), [1, 1, 1, 1, 1, 1, 1]))
        game.step(0)
        game.step(0)
        game.step(0)
        game.step(0)
        game.step(0)
        self.assertTrue(np.array_equal(
            game.availableMoves(), [1, 1, 1, 1, 1, 1, 1]))
        game.step(0)
        self.assertTrue(np.array_equal(
            game.availableMoves(), [0, 1, 1, 1, 1, 1, 1]))


if __name__ == '__main__':
    unittest.main()
