import numpy as np
from const import *
import copy


class Hand():
    def __init__(self):
        self.a = 3
        self.g = 3
        self.s = 2
        self.b = 2
        self.q = 1


class AvalilableActions():
    def __init__(self):
        self.placeActions = []
        self.moveActions = []

    def empty(self):
        return not self.placeActions and not self.moveActions

    def addPlaceAction(self, piece, x, y):
        self.placeActions.append(PlaceAction(piece, x, y))

    def addMoveAction(self, startX, startY, endX, endY):
        self.moveActions.append(MoveAction(startX, startY, endX, endY))

    def getPlaceActionsByPiece(self, piece):
        return [action for action in self.placeActions if action.piece == piece]

    def getMoveActionsByStart(self, startX, startY):
        return [action for action in self.moveActions if action.startX == startX and action.startY == startY]

    def getPlaceAction(self, piece, x, y):
        actions = [action for action in self.placeActions if action.piece == piece
                   and action.x == x and action.y == y]
        if len(actions) == 0:
            return None
        return actions[0]

    def getMoveAction(self, startX, startY, endX, endY):
        actions = [action for action in self.moveActions if action.startX == startX
                   and action.startY == startY and action.endX == endX and action.endY == endY]
        if len(actions) == 0:
            return None
        return actions[0]


class GameState():
    def __init__(self, board=np.zeros((23, 23), dtype="int"), player=-1, player1Hand=Hand(), player2Hand=Hand()):
        self.board = board
        self.player = player
        self.player1Hand = player1Hand
        self.player2Hand = player2Hand

        self.availableActions = self._availableActions()
        self.terminal, self.terminalValue = self._over()

        if self.availableActions.empty() and not self.terminal:  # if player blocked this turn
            self.player *= -1
            self.availableActions = self._availableActions()

    def _over(self):
        return False, None

    def _availableActions(self):
        avalilableActions = AvalilableActions()
        if np.count_nonzero(self.board) == 0:
            avalilableActions.addPlaceAction(Player1Q, 11, 11)
        avalilableActions.addMoveAction(11, 11, 12, 12)
        return avalilableActions


class PlaceAction():
    def __init__(self, piece, x, y):
        self.piece = piece
        self.x = x
        self.y = y

    def do(self, state: GameState):
        state = copy.deepcopy(state)
        if(self.piece == Player1A):
            state.player1Hand.a -= 1
        elif(self.piece == Player1G):
            state.player1Hand.g -= 1
        elif(self.piece == Player1S):
            state.player1Hand.s -= 1
        elif(self.piece == Player1B):
            state.player1Hand.b -= 1
        elif(self.piece == Player1Q):
            state.player1Hand.q -= 1
        elif(self.piece == Player2A):
            state.player2Hand.a -= 1
        elif(self.piece == Player2G):
            state.player2Hand.g -= 1
        elif(self.piece == Player2S):
            state.player2Hand.s -= 1
        elif(self.piece == Player2B):
            state.player2Hand.b -= 1
        elif(self.piece == Player2Q):
            state.player2Hand.q -= 1
        state.board[self.x][self.y] = self.piece
        state.player *= -1
        return state


class MoveAction():
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

    def do(self, state: GameState):
        state = copy.deepcopy(state)
        state.board[self.endX][self.endY] = state.board[self.startX][self.startY]
        state.board[self.startX][self.startY] = 0
        state.player *= -1
        return state


class Hive():
    def newGame(self):
        return GameState()

    def step(self, state: GameState, action):
        # raise Exception(f'Invalid action {action}')
        return action.do(state)
