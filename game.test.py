from game import Game
import unittest
import numpy as np


class TestStringMethods(unittest.TestCase):

    def test_play(self):
        game = Game()
        expected_board = np.zeros((6, 7), dtype="int")
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), 0)
        self.assertEqual(game.current_player, -1)

        self.assertFalse(game.step(0))
        expected_board[0][0] = -1
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), 1)
        self.assertEqual(game.current_player, 1)

        expected_board[1][0] = 1
        self.assertFalse(game.step(0))
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), 2)
        self.assertEqual(game.current_player, -1)

        expected_board[2][0] = -1
        self.assertFalse(game.step(0))
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), 3)
        self.assertEqual(game.current_player, 1)

        expected_board[3][0] = 1
        self.assertFalse(game.step(0))
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), 4)
        self.assertEqual(game.current_player, -1)

        expected_board[4][0] = -1
        self.assertFalse(game.step(0))
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), 5)
        self.assertEqual(game.current_player, 1)

        expected_board[5][0] = 1
        self.assertFalse(game.step(0))
        self.assertTrue(np.array_equal(game.board, expected_board))
        self.assertEqual(game.get_free_row(0), None)
        self.assertEqual(game.current_player, -1)

        self.assertRaises(Exception, game.step, 0)

    def test_step_reset(self):
        game = Game()
        self.assertRaises(Exception, game.step, -1)
        self.assertRaises(Exception, game.step, 7)

        self.assertFalse(game.step(0))
        self.assertFalse(game.step(6))
        self.assertFalse(game.step(6))

        expected_board = np.zeros((6, 7), dtype="int")
        expected_board[0][0] = -1
        expected_board[0][6] = 1
        expected_board[1][6] = -1

        self.assertEqual(game.current_player, 1)
        self.assertTrue(np.array_equal(game.board, expected_board))

        game.reset()

        expected_board = np.zeros((6, 7), dtype="int")
        self.assertEqual(game.current_player, -1)
        self.assertTrue(np.array_equal(game.board, expected_board))

    def test_done(self):
        game = Game()
        self.assertFalse(game.step(0))
        self.assertFalse(game.step(1))
        self.assertFalse(game.step(0))
        self.assertFalse(game.step(1))
        self.assertFalse(game.step(0))
        self.assertFalse(game.step(1))
        self.assertTrue(game.step(0))
        self.assertEqual(game.check_game_over(), -1)

        game = Game()
        game.board = np.ones((6, 7), dtype="int")
        self.assertEqual(game.check_game_over(), 0)

        game.board = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype="int")
        self.assertEqual(game.check_game_over(), None)

        game.board = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype="int")
        self.assertEqual(game.check_game_over(), 1)

        game.board = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, -1, 0],
             [0, 0, 1, 0, -1, 0, 0],
             [0, 0, 0, -1, 0, 0, 0],
             [0, 0, -1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]], dtype="int")
        self.assertEqual(game.check_game_over(), -1)

        game.board = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 0, 0]], dtype="int")
        self.assertEqual(game.check_game_over(), 1)

        game.board = np.array(
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, -1, -1, -1, -1]], dtype="int")
        self.assertEqual(game.check_game_over(), -1)


if __name__ == '__main__':
    unittest.main()
