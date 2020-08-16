import numpy as np
from Connect4Game import Connect4Game
from TTT import TTTGame
from Hive.Net import Net
import random
from MCTS import MCTS
from TTTNet.NNetWrapper import NNetWrapper as NNet

game = TTTGame()
net = NNet()
net.load_checkpoint()

mcts = MCTS(game, net)




for _ in range(1):
    trainingExamples = []
    for _ in range(10):
        state = game.newGame()
        foo = []
        while True:
            root = mcts.getPolicy(50, state, False)
            pi = mcts.getPolicy(50, state, False)
            action = np.random.choice(np.arange(len(pi)), p=pi)
            action = np.argmax(pi)
            foo.append([state, pi])
            # add (state,policy,reward=None) to trainingExamples
            state = game.step(state, action)
            if state.terminal:
                for f in foo:
                    f.append(state.terminalValue)
                print('gameOver', state.terminalValue)
                break
        trainingExamples.append(foo)
    print(trainingExamples)
    # trainNet(trainingExamples)
