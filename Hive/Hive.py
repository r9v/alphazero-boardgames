import numpy as np


class Hive():
    def newGame(self):
        # returns board, currentPlayer
        return np.zeros((23, 23), dtype="int"), Hand(), -1


class Hand():
    def __init__(self):
        self.a = 3
        self.g = 3
        self.s = 2
        self.b = 2
        self.q = 1
