import random
import math
import numpy as np


class Node():
    def __init__(self, parent, state, game, nnet):
        self.parent = parent
        self.state = state
        # dictionary i->node. i is the action avalilableActions[i]
        self.children = {}

        if not self.state.terminal:
            self.availableActionsMask = state.availableActions
        else:
            self.availableActionsMask = []
        self.availableActions = np.nonzero(self.availableActionsMask)[0]
        for action in self.availableActions:
            self.children[action] = None

        self.n = 0
        self.Q = 0.0
        self.W = 0.0

        # P, v = nnet.predict()
        self.P = [0, 0, 0, 0.5, 0.6, 0.7, 0, 0, 0]
        self.nnetValue = random.randint(-40, 40)


class MCTS():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet

    def getPolicy(self, numMCTSSimulations, state) -> Node:
        if state.terminal:
            raise Exception('Called getPolicy with terminal state')

        root = Node(None, state, self.game, self.nnet)
        for i in range(numMCTSSimulations):
            self.search(root)
        return root
        # calc policy for root node

    def search(self, root: Node):
        selectedNode = self.treePolicy(root)
        value = self.rollout(selectedNode)
        self.backpropagate(value, selectedNode)

    def treePolicy(self, node: Node):
        while not node.state.terminal:
            bestAction = self.bestAction(node)
            print(f'bestAction {bestAction}')
            if node.children[bestAction] is None:
                return self.createChild(node, bestAction)
            node = node.children[bestAction]
        return node

    def bestAction(self, node: Node):
        if(node.n == 0):
            return np.argmax(np.multiply(
                node.P, node.availableActionsMask))

        bestPUCT = -float('inf')
        bestAction = None
        for availableAction in node.availableActions:
            child = node.children[availableAction]
            Q, N = 0.0, 0
            if child is not None:
                Q = -child.Q
                N = child.n
            actionPUCT = self.PUCT(Q, node.P[availableAction], N, node.n)
            if(actionPUCT > bestPUCT):
                bestPUCT = actionPUCT
                bestAction = availableAction
        return bestAction

    def createChild(self, node, action):
        node.children[action] = Node(
            node, self.game.step(node.state, action), self.game, self.nnet)
        return node.children[action]

    def PUCT(self, Q, P, N, Nparent):
        return Q+P*math.sqrt(Nparent)/(N+1)

    def rollout(self, node: Node):
        if node.state.terminal:
            return node.state.terminalValue
        return node.nnetValue

    def backpropagate(self, value, node: Node):
        while node is not None:
            node.n += 1
            node.W += value
            node.Q = node.W/node.n
            node = node.parent
            value = -value
