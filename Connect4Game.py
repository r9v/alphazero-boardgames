import numpy as np
from collections import namedtuple

ROW_COUNT = 6
COLUMN_COUNT = 7

GameEndedState = namedtuple('GameEndedState', 'done victor')


class Connect4Game():
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
        self.currentPlayer = -1

    def step(self, column):
        if(column < 0 or column > COLUMN_COUNT-1):
            raise Exception(f'Invalid column {column}')

        row = self.getFirstFreeRow(column)
        if row is None:
            raise Exception(f'Invalid action, column {column} is full')
        self.board[row][column] = self.currentPlayer
        self.currentPlayer *= -1

        gameEndedState = self.checkGameEnded()
        if gameEndedState.done:
            if gameEndedState.victor is not None:
                return True, 0, self.board
            return True, gameEndedState.victor, self.board
        return False, 0, self.board

    def getFirstFreeRow(self, column):
        for row in range(ROW_COUNT):
            if self.board[row][column] == 0:
                return row
        return None

    def availableMoves(self):
        availableMoves = []
        for column in range(COLUMN_COUNT):
            if self.getFirstFreeRow(column) is not None:
                availableMoves.append(column)
        return availableMoves

    def print(self):
        print(np.flip(self.board, 0))

    def checkPlayerWon(self, player):
        for c in range(COLUMN_COUNT-3):  # Horizontal
            for r in range(ROW_COUNT):
                if self.board[r][c] == player and self.board[r][c+1] == player and self.board[r][c+2] == player and self.board[r][c+3] == player:
                    return True
        for c in range(COLUMN_COUNT):  # Vertical
            for r in range(ROW_COUNT-3):
                if self.board[r][c] == player and self.board[r+1][c] == player and self.board[r+2][c] == player and self.board[r+3][c] == player:
                    return True
        for c in range(COLUMN_COUNT-3):  # Diag1
            for r in range(ROW_COUNT-3):
                if self.board[r][c] == player and self.board[r+1][c+1] == player and self.board[r+2][c+2] == player and self.board[r+3][c+3] == player:
                    return True
        for c in range(COLUMN_COUNT-3):  # Diag2
            for r in range(3, ROW_COUNT):
                if self.board[r][c] == player and self.board[r-1][c+1] == player and self.board[r-2][c+2] == player and self.board[r-3][c+3] == player:
                    return True

    def checkGameEnded(self):
        if np.count_nonzero(self.board) == ROW_COUNT*COLUMN_COUNT:
            return GameEndedState(True, None)
        if(self.checkPlayerWon(-1)):
            return GameEndedState(True, -1)
        if(self.checkPlayerWon(1)):
            return GameEndedState(True, 1)
        return GameEndedState(False, None)
