import numpy as np
from const import *


class PlaceAction():
    def __init__(self, piece, x, y):
        self.piece = piece
        self.x = x
        self.y = y


class AvalilableActions():
    def __init__(self):
        self.placeActions = []
        self.moveActions = []

    def addPlaceAction(self, piece, x, y):
        self.placeActions.append(PlaceAction(piece, x, y))

    def getPlaceActionsByPiece(self, piece):
        return [placeAction for placeAction in self.placeActions if placeAction.piece == piece]

    def isPlaceActionCorrect(self, piece, x, y):
        return len([placeAction for placeAction in self.placeActions if placeAction.piece == piece
                    and placeAction.x == x and placeAction.y == y]) != 0


class Hive():
    def newGame(self):
        # returns board, player1 hand, player2 hand, currentPlayer
        return np.zeros((23, 23), dtype="int"), Hand(), Hand(), -1

    def availableActions(self, board, playersHand, currentPlayer):
        avalilableActions = AvalilableActions()
        if np.count_nonzero(board) == 0:
            avalilableActions.addPlaceAction(Player1Q, 11, 11)
        return avalilableActions


class Hand():
    def __init__(self):
        self.a = 3
        self.g = 3
        self.s = 2
        self.b = 2
        self.q = 1
