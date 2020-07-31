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
        board, currentPlayer = game.reset()
        while True:
            policy = mcts.getPolicy(NUM_MCTS_SIMULATIONS, board, currentPlayer)
            # get action from pi
            # add (state,policy,reward=None) to trainingExamples
            done, reward, board, currentPlayer = game.step(action)
            if done:
                # set reward for all trainingExamples
                break
    trainNet(trainingExamples)
