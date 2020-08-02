from Connect4Game import Connect4Game
from TTT import TTTGame
import random
import numpy as np

game = TTTGame()


class Node():
    def __init__(self, parent, board, player, nnet):
        self.parent = parent

        # dictionary i->node. i is the action avalilableActions[i]
        self.children = {}

        self.terminal, self.terminalValue = game.over(board)
        if not self.terminal:
            self.availableActionsMask = game.availableActions(board, player)
        else:
            self.availableActionsMask = []
        self.availableActions = np.nonzero(self.availableActionsMask)[0]
        for action in self.availableActions:
            self.children[action] = None

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
            node = bestChild(node)
            if node.n == 0:
                return node
        return node

    def bestChild(self, node: Node):
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
            actionPUCT = PUCT(Q, node.P[availableAction], N, node.n)
            if(actionPUCT > bestPUCT):
                bestPUCT = actionPUCT
                bestAction = availableAction

        if node.children[bestAction] is None:
            node.children[bestAction] = Node(node, newBoard, newPlayer, nnet)

        return node.children[bestAction]

    def PUCT(self, Q, P, N, Nparent):
        return Q+P*sqrt(Nparent)/(N+1)

    def rollout(self, node: Node):
        if node.terminal:
            return node.terminalValue
        return node.nnetValue

    def backpropagate(self, value, node: Node):
        while node is not None:
            node.n += 1
            node.Q = (node.Q+value)/node.n
            node = node.parent
