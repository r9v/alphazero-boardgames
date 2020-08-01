import numpy as np
from collections import namedtuple

ROW_COUNT = 6
COLUMN_COUNT = 7


class Connect4Game():

    def newGame(self):
        # returns board, currentPlayer
        return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int"), -1

    def step(self, board, currentPlayer, column):
        # returns done, victor(or reward - 0 if draw), board, currentPlayer
        if(column < 0 or column > COLUMN_COUNT-1):
            raise Exception(f'Invalid column {column}')

        row = self._getFirstFreeRow(board, column)
        if row is None:
            raise Exception(f'Invalid action, column {column} is full')
        board[row][column] = currentPlayer
        currentPlayer *= -1

        return board, currentPlayer

    def _getFirstFreeRow(self, board, column):
        for row in range(ROW_COUNT):
            if board[row][column] == 0:
                return row
        return None

    def availableMoves(self, board):
        availableMoves = [0]*COLUMN_COUNT
        for column in range(COLUMN_COUNT):
            if self._getFirstFreeRow(column, board) is not None:
                availableMoves[column] = 1
        return availableMoves

    def _checkPlayerWon(self, board, player):
        for c in range(COLUMN_COUNT-3):  # Horizontal
            for r in range(ROW_COUNT):
                if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                    return True
        for c in range(COLUMN_COUNT):  # Vertical
            for r in range(ROW_COUNT-3):
                if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                    return True
        for c in range(COLUMN_COUNT-3):  # Diag1
            for r in range(ROW_COUNT-3):
                if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                    return True
        for c in range(COLUMN_COUNT-3):  # Diag2
            for r in range(3, ROW_COUNT):
                if board[r][c] == player and board[r-1][c+1] == player and board[r-2][c+2] == player and board[r-3][c+3] == player:
                    return True

    def over(self, board):
      # return gameOver, victor(reward - 0 if tie)
        if np.count_nonzero(board) == ROW_COUNT*COLUMN_COUNT:
            return True, 0
        if(self._checkPlayerWon(board, -1)):
            return True, -1
        if(self._checkPlayerWon(board, 1)):
            return True, 1
        return False, None
