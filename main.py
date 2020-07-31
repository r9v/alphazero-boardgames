import numpy as np
from Connect4Game import Connect4Game
import random

NUM_ITERS = 2
NUM_EPISODES = 2
NUM_MCTS_SIMULATIONS = 2
game = Connect4Game()

for _ in range(NUM_ITERS):
    trainingExamples = []
    for _ in range(NUM_EPISODES):
        mcts = MCTS()
        while True:
            pi = mcts.getPolicy(NUM_MCTS_SIMULATIONS)
            # get action from pi
            done, reward, board, currentPlayer = game.step(action)
            if done:
                break
    trainNet(trainingExamples)
