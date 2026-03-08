import math
import numpy as np


def add_dirichlet_noise(arr, alpha, epsilon):
    noise = np.random.dirichlet(np.ones(len(arr)) * alpha)
    return arr * (1 - epsilon) + epsilon * noise


class Node:
    def __init__(self, parent, state, game, net):
        self.parent = parent
        self.state = state
        self.children = {}

        if not self.state.terminal:
            self.available_actions_mask = state.available_actions
        else:
            self.available_actions_mask = []
        self.available_actions = np.nonzero(self.available_actions_mask)[0]
        for action in self.available_actions:
            self.children[action] = None

        self.n = 0
        self.Q = 0.0
        self.W = 0.0

        state_input = game.state_to_input(state)
        self.nnet_value, self.P = net.predict(state_input)


class MCTS:
    def __init__(self, game, net):
        self.game = game
        self.net = net

    def get_policy(self, num_simulations, state, add_dirichlet=False):
        if state.terminal:
            raise ValueError("Called get_policy with terminal state")

        root = Node(None, state, self.game, self.net)
        if add_dirichlet:
            root.P = add_dirichlet_noise(root.P, 0.03, 0.25)
        for _ in range(num_simulations):
            self._search(root)

        pi = np.zeros(np.shape(root.available_actions_mask))
        for action in root.available_actions:
            if root.children[action] is not None:
                pi[action] = root.children[action].n / root.n
        return pi

    def _search(self, root):
        selected = self._tree_policy(root)
        value = self._evaluate(selected)
        self._backpropagate(value, selected)

    def _tree_policy(self, node):
        while not node.state.terminal:
            best = self._best_action(node)
            if node.children[best] is None:
                return self._create_child(node, best)
            node = node.children[best]
        return node

    def _best_action(self, node):
        if node.n == 0:
            return np.argmax(np.multiply(node.P, node.available_actions_mask))

        best_puct = -float('inf')
        best_action = None
        for action in node.available_actions:
            child = node.children[action]
            q, n = 0.0, 0
            if child is not None:
                q = child.Q
                n = child.n
            puct = q + node.P[action] * math.sqrt(node.n) / (n + 1)
            if puct > best_puct:
                best_puct = puct
                best_action = action
        return best_action

    def _create_child(self, node, action):
        child = Node(node, self.game.step(node.state, action), self.game, self.net)
        node.children[action] = child
        return child

    def _evaluate(self, node):
        if node.state.terminal:
            return -node.state.terminal_value * node.state.player
        return -node.nnet_value * node.state.player

    def _backpropagate(self, value, node):
        while node is not None:
            if node.state.last_turn_skipped:
                value = -value
            node.n += 1
            node.W += value
            node.Q = node.W / node.n
            node = node.parent
            value = -value
