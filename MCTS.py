from Connect4Game import Connect4Game
from TTT import TTTGame
import random

game = TTTGame()


class Node():
    def __init__(self, parent, board, player):
        self.parent = parent
        self.children = []

        self.n = 0
        self.Q = 0

        self.board = board
        self.player = player

    def getChildren(self):
        f = 1


class MCTS():

    def getPolicy(self, numMCTSSimulations, board, player):
        over, _ = game.over(board)
        if over:
            raise Exception('Called getPolicy with over game')

        root = Node(None, board, player)
        for i in range(numMCTSSimulations):
            self.search(root)

        # calc policy for root node

    def search(self, root):
        # if(root.ifFinalState)
        # return 1000
        selectedNode = select(root)
        if selectedNode is None:
            return
        value = None
        if selectedNode.n == 0:
            value = rollout(selectedNode)
        else:
            if selectedNode.n == 1:
                selectedNode.getChildren()
            value = search(selectedNode)
        backpropagate(value, selectedNode)

    def select(self, node: Node):
        bestUCB = -float('inf')
        bestChild = None

        for child in node.children:
            if child.n == 0:
                return child

        for child in node.children:
            childUCB = UCB(child)
            if(childUCB > bestUCB):
                bestUCB = childUCB
                bestChild = child
        return bestChild

    def rollout(self, node: Node):
        return random.randint(-40, 40)

    def backpropagate(self, value, node: Node):
        while node is not None:
            node.n += 1
            node.Q = (node.Q+value)/node.n
            node = node.parent

    def expand(self, node):
        node.getChildren()
