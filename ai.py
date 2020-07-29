import numpy as np
from game import Game
import random

ROW_COUNT = 6
COLUMN_COUNT = 7

done = False
game = Game()

while not done:
    move = random.choice(game.available_moves())
    done = game.step(move)
    game.print()
