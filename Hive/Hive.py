import numpy as np
from const import *


class Hive():
    def newGame(self):
        # returns board, player1 hand, player2 hand, currentPlayer
        return np.zeros((23, 23), dtype="int"), Hand(), Hand(), -1

    def availableActions(self, board, playersHand, currentPlayer):
        avalilableActions = {'place': {}}
        if np.count_nonzero(board) == 0:
            avalilableActions['place'] = {Player1Q: [(11, 11)]}
        return avalilableActions


class Hand():
    def __init__(self):
        self.a = 3
        self.g = 3
        self.s = 2
        self.b = 2
        self.q = 1
