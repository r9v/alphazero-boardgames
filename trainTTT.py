import numpy as np
import random
from TTT.TTT import TTTGame
from TTT.Net import Net
from MCTS import MCTS
from TrainingData import TrainingData

game = TTTGame()
net = Net()
mcts = MCTS(game, net)


trainingData = TrainingData(100)
MINIBATCH_SIZE = 2
GAMES_PER_AGENT = 2

for _ in range(1):
    for _ in range(GAMES_PER_AGENT):
        state = game.newGame()
        trainingDataFromGame = []
        while True:
            pi = mcts.getPolicy(50, state, False)
            action = np.random.choice(np.arange(len(pi)), p=pi)
            # action = np.argmax(pi)
            trainingDataFromGame.append([state, pi])
            state = game.step(state, action)
            if state.terminal:
                for d in trainingDataFromGame:
                    d.append(state.terminalValue)
                print('gameOver', state.terminalValue)
                break
        trainingData.insertArr(trainingDataFromGame)
    print(trainingData.arr)
    # trainNet(trainingData)
    # netDuel
