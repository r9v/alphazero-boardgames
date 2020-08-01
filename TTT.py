import numpy as np
from collections import namedtuple

ROW_COUNT = 6
COLUMN_COUNT = 7


class TTTGame():

    def newGame(self):
        # returns board, currentPlayer
        return np.zeros((3, 3), dtype="int"), -1

    def step(self, board, currentPlayer, action):
        # returns done, victor(or reward - 0 if draw), board, currentPlayer
        if(action < 0 or action > 8):
            raise Exception(f'Invalid action {action}')

        x = action//3
        y = action % 3
        if board[x][y] != 0:
            raise Exception(f'Invalid action, {x},{y} is full')
        nextBoard = np.copy(board)
        nextBoard[x][y] = currentPlayer

        return nextBoard, currentPlayer * -1

    def over(self, board):
        if np.count_nonzero(board) == 9:
            return True, 0
        if(self._checkPlayerWon(board, -1)):
            return True, -1
        if(self._checkPlayerWon(board, 1)):
            return True, 1
        return False, None

    def _checkPlayerWon(self, board, player):
        for row in range(3):
            if board[row][0] == player and board[row][1] == player and board[row][2] == player:
                return True

        for col in range(3):
            if board[0][col] == player and board[1][col] == player and board[2][col] == player:
                return True

        if board[0][0] == player and board[1][1] == player and board[2][2] == player:
            return True

        if board[0][2] == player and board[1][1] == player and board[2][0] == player:
            return True
