import numpy as np
import random
import pickle

from TTT.TTT import TTTGame
from TTT.Net import Net
from MCTS import MCTS
from TrainingData import TrainingData



MINIBATCH_SIZE = 2
GAMES_PER_AGENT = 2
def loadLastNetworkAndData(startBlank):
    net = Net()
    if startBlank:
        return net, TrainingData()
    net.loadLatest()
    trainingData = None
    with open(r"TTT/trainingDataSave.pickle", "rb") as input_file:
        trainingData = pickle.load(input_file)
    return net, trainingData



game = TTTGame()
net, trainingData = loadLastNetworkAndData(True)
mcts = MCTS(game, net)
for _ in range(1):
    for _ in range(GAMES_PER_AGENT):
        state = game.newGame()
        trainingDataFromGame = []
        while True:
            pi = mcts.getPolicy(50, state, False)
            action = np.random.choice(np.arange(len(pi)), p=pi)
            # action = np.argmax(pi)
            trainingDataFromGame.append([net.stateToInput(state), pi])
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
