from Connect4Game import Connect4Game
from TTT import TTTGame
import random
import numpy as np

game = TTTGame()


class Node():
    def __init__(self, parent, board, player, nnet):
        self.parent = parent
        self.children = []

        gameOver, _ = game.over(board)
        self.terminal = gameOver
        if not self.terminal:
            self.availableActionsMask = game.availableActions(board, player)
        else:
            self.availableActionsMask = []
        self.availableActions = np.nonzero(self.availableActionsMask)[0]

        self.n = 0
        self.Q = 0.0

        P, v = nnet.predict()
        self.P = [0, 0, 0, 0.5, 0.6, 0.7, 0, 0, 0]
        self.nnetValue = random.randint(-40, 40)

        self.board = board
        self.player = player


class MCTS():

    def getPolicy(self, numMCTSSimulations, board, player):
        gameOver, _ = game.over(board)
        if gameOver:
            raise Exception('Called getPolicy with gameOver')

        root = Node(None, board, player)
        for i in range(numMCTSSimulations):
            self.search(root)

        # calc policy for root node

    def search(self, root: Node):
        selectedNode = treePolicy(root)
        value = rollout(selectedNode)
        backpropagate(value, selectedNode)

    def treePolicy(self, node: Node):
        while not node.terminal:
            node, neverVisited = bestChild(node)
            if neverVisited:
                return node
        return node

    def bestChild(self, node: Node):
        bestPUCT = -float('inf')
        bestAction = None
        for idx, availableAction in enumerate(node.availableActions):
            child = parent.children[availableAction]
            if child is not None:
                Q = child.Q
                N = child.n
            else:
                Q = 0
                N = 0
            actionPUCT = PUCT(child, Q, node.P, N, node.n)
            if(actionPUCT > bestPUCT):
                bestPUCT = actionPUCT
                bestAction = availableAction

        return bestChild

    def PUCT(self, Q, P, N, Nparent):
        return Q+P*sqrt(Nparent)/(N+1)

    def rollout(self, node: Node):
        # if state terminal return realValue
        return node.nnetValue

    def backpropagate(self, value, node: Node):
        while node is not None:
            node.n += 1
            node.Q = (node.Q+value)/node.n
            node = node.parent
