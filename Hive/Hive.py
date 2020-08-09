import numpy as np
from const import *


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

    def addPlaceAction(self, piece, x, y):
        self.placeActions.append(PlaceAction(piece, x, y))

    def addMoveAction(self, startX, startY, endX, endY):
        self.moveActions.append(MoveAction(startX, startY, endX, endY))

    def getPlaceActionsByPiece(self, piece):
        return [action for action in self.placeActions if action.piece == piece]

    def isPlaceActionCorrect(self, piece, x, y):
        return len([action for action in self.placeActions if action.piece == piece
                    and action.x == x and action.y == y]) != 0

    def isMoveActionCorrect(self, startX, startY, endX, endY):
        return len([action for action in self.moveActions if action.startX == startX
                    and action.startY == startY and action.endX == endX and action.endY == endY]) != 0

    def getMoveActionsByStart(self, startX, startY):
        return [action for action in self.moveActions if action.startX == startX and action.startY == startY]


class GameState():
    def __init__(self, board=np.zeros((23, 23), dtype="int"), player=-1, player1Hand=Hand(), player2Hand=Hand()):
        self.board = board
        self.player = player
        self.player1Hand = player1Hand
        self.player2Hand = player2Hand

        self.availableActions = self._availableActions()
        self.terminal, self.terminalValue = self._over()

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
        return GameState(state.board, state.player, state.player1Hand, state.player2Hand)


class MoveAction():
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

    def do(self, state: GameState):
        return GameState(state.board, state.player, state.player1Hand, state.player2Hand)


class Hive():
    def newGame(self):
        return GameState()

    def step(self, state: GameState, action):
        # raise Exception(f'Invalid action {action}')
        return action.do(state)
