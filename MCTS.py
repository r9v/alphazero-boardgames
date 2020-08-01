from Connect4Game import Connect4Game
from TTT import TTTGame
import random

game = TTTGame()


class Node():
    def __init__(self, parent, board, player):
        self.parent = parent
        self.children = []

        gameOver, _ = game.over(board)
        self.terminal = gameOver
        if not gameOver:
            self.availableActions = game.availableActions(board, player)
        else:
            self.availableActions = []
        self.n = 0
        self.Q = 0

        self.board = board
        self.player = player

    def fullyExpanded(self):
        len(self.children) == np.count_nonzero(self.availableActions)


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
            if not node.fullyExpanded():
                return expand(node)
            node = bestChild(node)
        return node

    def expand(self, node: Node):
        action = node.availableActions(len(node.children)+1)
        newChild = 1  # make child from action
        node.children.append(newChild)
        return newChild

    def bestChild(self, node: Node):
        bestUCB = -float('inf')
        bestChild = None
        for child in node.children:
            childUCB = UCB(child)
            if(childUCB > bestUCB):
                bestUCB = childUCB
                bestChild = child
        return bestChild

    def rollout(self, node: Node):
        return random.randint(-40, 40)
        # get value from NNet

    def backpropagate(self, value, node: Node):
        while node is not None:
            node.n += 1
            node.Q = (node.Q+value)/node.n
            node = node.parent
