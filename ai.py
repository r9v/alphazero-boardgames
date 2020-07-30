import numpy as np
from game import Connect4Game
import random

ROW_COUNT = 6
COLUMN_COUNT = 7

done = False
game = Connect4Game()

while not done:
    move = random.choice(game.availableMoves())
    done, score, board = game.step(move)
    print(done)
    print(score)
    print(board)
    game.print()
