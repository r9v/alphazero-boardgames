import numpy as np
from collections import namedtuple

ROW_COUNT = 6
COLUMN_COUNT = 7


class GameState():
    def __init__(self, board=np.zeros((3, 3), dtype="int"), player=-1):
        self.board = board
        self.player = player
        self.availableActions = self._availableActions()
        self.terminal, self.terminalValue = self._over()
        self.lastTurnSkipped = False

    def _over(self):
        if(self._checkPlayerWon(self.board, -1)):
            return True, -1
        if(self._checkPlayerWon(self.board, 1)):
            return True, 1
        if np.count_nonzero(self.board) == 9:
            return True, 0
        return False, None

    def _availableActions(self):
        availableActions = [0]*9
        for row in range(3):
            for column in range(3):
                if self.board[row][column] == 0:
                    availableActions[3*row+column] = 1
        return availableActions

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


class TTTGame():

    def newGame(self):
        return GameState()

    def step(self, state, action):
        if(action < 0 or action > 8):
            raise Exception(f'Invalid action {action}')

        x = action//3
        y = action % 3
        if state.board[x][y] != 0:
            raise Exception(f'Invalid action, {x},{y} is full')
        nextBoard = np.copy(state.board)
        nextBoard[x][y] = state.player

        funBoard = np.zeros((3, 3), dtype="int")
        funBoard[0][0] = 1
        funBoard[0][1] = 0
        funBoard[0][2] = 1

        funBoard[1][0] = 0
        funBoard[1][1] = 0
        funBoard[1][2] = -1

        funBoard[2][0] = 0
        funBoard[2][1] = 0
        funBoard[2][2] = -1
        astate = GameState(nextBoard, state.player * -1)
        if action == 4 and np.array_equal(state.board, funBoard):
            astate.player = astate.player*-1
            astate.lastTurnSkipped = True
        return astate
