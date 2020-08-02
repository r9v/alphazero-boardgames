from Connect4Game import Connect4Game
from TTT import TTTGame
import random
import math
import numpy as np


class Node():
    def __init__(self, parent, board, player, game, nnet):
        self.parent = parent

        # dictionary i->node. i is the action avalilableActions[i]
        self.children = {}

        self.terminal, self.terminalValue = game.over(board)
        if not self.terminal:
            self.availableActionsMask = game.availableActions(board, player)
        else:
            self.availableActionsMask = []
        self.availableActions = np.nonzero(self.availableActionsMask)[0]
        print(f'self.availableActionsMask {self.availableActionsMask}')
        print(f'self.availableActions {self.availableActions}')
        for action in self.availableActions:
            self.children[action] = None

        self.n = 0
        self.Q = 0.0

        # P, v = nnet.predict()
        self.P = [0, 0, 0, 0.5, 0.6, 0.7, 0, 0, 0]
        self.nnetValue = random.randint(-40, 40)

        self.board = board
        self.player = player

    def print(self):
        print(f'board')


class MCTS():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet

    def getPolicy(self, numMCTSSimulations, board, player) -> Node:
        gameOver, _ = self.game.over(board)
        if gameOver:
            raise Exception('Called getPolicy with gameOver')

        root = Node(None, board, player, self.game, self.nnet)
        for i in range(numMCTSSimulations):
            self.search(root)
        return root
        # calc policy for root node

    def search(self, root: Node):

        selectedNode = self.treePolicy(root)
        value = self.rollout(selectedNode)
        print(f'value {value}')
        self.backpropagate(value, selectedNode)

    def treePolicy(self, node: Node):
        while not node.terminal:
            bestAction = self.bestAction(node)
            if node.children[bestAction] is None:
                self.createChild(node, bestAction)
            node = node.children[bestAction]
            if node.n == 0:
                return node
        return node

    def bestAction(self, node: Node):
        if(node.n == 0):
            return np.argmax(np.multiply(
                node.P, node.availableActionsMask))

        bestPUCT = -float('inf')
        bestAction = None
        for idx, availableAction in enumerate(node.availableActions):
            child = node.children[availableAction]
            Q, N = 0.0, 0
            if child is not None:
                Q = child.Q
                N = child.n
            else:
                Q = 0.0
                N = 0
            actionPUCT = self.PUCT(Q, node.P[availableAction], N, node.n)
            if(actionPUCT > bestPUCT):
                bestPUCT = actionPUCT
                bestAction = availableAction
        return bestAction

    def createChild(self, node, action):
        newBoard, newPlayer = self.game.step(node.board, node.player, action)
        node.children[action] = Node(
            node, newBoard, newPlayer, self.game, self.nnet)

    def PUCT(self, Q, P, N, Nparent):
        return Q+P*math.sqrt(Nparent)/(N+1)

    def rollout(self, node: Node):
        if node.terminal:
            return node.terminalValue
        return node.nnetValue

    def backpropagate(self, value, node: Node):
        while node is not None:
            node.n += 1
            node.Q = (node.Q+value)/node.n
            node = node.parent
