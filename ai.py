import numpy as np
from game import Game

ROW_COUNT = 6
COLUMN_COUNT = 7

done = False
game = Game()

while not done:
    move = game.available_moves()[0]
    # return obs, rewards, game_over, {}

    obs, reward, done = con4.step(move)
    print(con4.available_moves())
