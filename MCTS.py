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
        gameOver, _ = game.over(board)
        self.gameOver = gameOver

    def getChildren(self):
        f = 1


class MCTS():

    def getPolicy(self, numMCTSSimulations, board, player):
        root = Node(None, board, player)
        root.getChildren()
        for i in range(numMCTSSimulations):
            self.search(tree)

        # calc policy for root node

    def search(self, root):
        selectedNode = select(root)
        if selectedNode is None:
            return
        value = None
        if selectedNode.n == 0:
            value = rollout(selectedNode)
        else:
            value = expand(selectedNode)
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
        if Node is None:
            return
        node.n += 1
        node.Q = (node.Q+value)/node.n
        self.backpropagate(value, node.parent)

    def expand(self, node):
        node.getChildren()
