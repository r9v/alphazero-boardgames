# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated MCTS with typed CNode.

Replaces Python MCTS with C-typed operations:
- CNode has typed int/double fields (no Python object overhead)
- PUCT loop uses C-level math (cached sqrt, typed iteration)
- Backpropagation uses typed tree walk
- Children stored as list (O(1) index) instead of dict
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()


cdef cnp.ndarray _dirichlet_noise(cnp.ndarray arr, double alpha, double epsilon):
    """Add Dirichlet noise to a policy array."""
    noise = np.random.dirichlet(np.ones(len(arr)) * alpha)
    return arr * (1.0 - epsilon) + epsilon * noise


cdef class CNode:
    """MCTS tree node with C-typed fields."""
    cdef public CNode parent
    cdef public int n
    cdef public double Q
    cdef public double W
    cdef public double nnet_value
    cdef public object P  # numpy array (policy prior)
    cdef public list children  # list[CNode|None], length=action_size
    cdef public object state  # game state reference
    cdef public bint is_terminal
    cdef public int terminal_value
    cdef public int player
    cdef public bint last_turn_skipped

    # Available actions as typed array
    cdef int _avail[512]  # indices of available actions (max 512)
    cdef public int _num_available
    cdef object _avail_mask  # original mask for compatibility

    def __init__(self, CNode parent, object state, object game, object net=None):
        self.parent = parent
        self.state = state
        self.n = 0
        self.Q = 0.0
        self.W = 0.0
        self.is_terminal = state.terminal
        self.terminal_value = state.terminal_value if state.terminal else 0
        self.player = state.player
        self.last_turn_skipped = state.last_turn_skipped

        cdef int action_size, i, count
        if not self.is_terminal:
            self._avail_mask = state.available_actions
            action_size = len(self._avail_mask)
            self.children = [None] * action_size

            # Build typed available actions index
            count = 0
            for i in range(action_size):
                if self._avail_mask[i]:
                    self._avail[count] = i
                    count += 1
            self._num_available = count
        else:
            self._avail_mask = np.array([], dtype=np.intc)
            self.children = []
            self._num_available = 0

        if net is not None:
            state_input = game.state_to_input(state)
            self.nnet_value, p = net.predict(state_input)
            self.P = np.ascontiguousarray(p, dtype=np.float32)
        else:
            self.nnet_value = 0.0
            self.P = None

    def resolve(self, double value, object policy):
        """Fill in deferred neural network evaluation."""
        self.nnet_value = value
        self.P = np.ascontiguousarray(policy, dtype=np.float32)

    @property
    def available_actions_mask(self):
        return self._avail_mask

    cdef object _make_avail_array(self):
        cdef int i
        result = np.empty(self._num_available, dtype=np.intc)
        for i in range(self._num_available):
            result[i] = self._avail[i]
        return result

    @property
    def available_actions(self):
        """Return numpy array of available action indices."""
        return self._make_avail_array()


cdef int _best_action(CNode node):
    """PUCT action selection — typed loop with numpy P access."""
    cdef int i, action, best_action
    cdef double puct, best_puct, q, sqrt_n, p_val
    cdef int child_n
    cdef CNode child

    # P=None guard: node created during multi-select, not yet evaluated
    if node.P is None:
        return node._avail[0]

    # Access P as flat buffer pointer for speed (works with any float dtype)
    cdef cnp.ndarray P_arr = <cnp.ndarray>node.P
    cdef char* P_data = P_arr.data
    cdef int P_itemsize = P_arr.itemsize
    cdef bint P_is_float32 = (P_itemsize == 4)

    if node.n == 0:
        # First visit: pick by prior × mask
        best_puct = -1e30
        best_action = node._avail[0]
        for i in range(node._num_available):
            action = node._avail[i]
            if P_is_float32:
                p_val = <double>(<float*>(P_data + action * 4))[0]
            else:
                p_val = (<double*>(P_data + action * 8))[0]
            if p_val > best_puct:
                best_puct = p_val
                best_action = action
        return best_action

    # PUCT selection
    sqrt_n = sqrt(<double>node.n)
    best_puct = -1e30
    best_action = node._avail[0]

    for i in range(node._num_available):
        action = node._avail[i]
        child = <CNode>node.children[action]
        if child is not None:
            q = child.Q
            child_n = child.n
        else:
            q = 0.0
            child_n = 0
        if P_is_float32:
            p_val = <double>(<float*>(P_data + action * 4))[0]
        else:
            p_val = (<double*>(P_data + action * 8))[0]
        puct = q + p_val * sqrt_n / (<double>child_n + 1.0)
        if puct > best_puct:
            best_puct = puct
            best_action = action

    return best_action


cdef void _backpropagate(double value, CNode node) noexcept:
    """Backpropagate value up the tree — typed traversal."""
    while node is not None:
        if node.last_turn_skipped:
            value = -value
        node.n += 1
        node.W += value
        node.Q = node.W / <double>node.n
        node = node.parent
        value = -value


cdef double _evaluate(CNode node) noexcept:
    """Get leaf node value."""
    if node.is_terminal:
        return -<double>node.terminal_value * <double>node.player
    return -node.nnet_value * <double>node.player


cdef void _apply_virtual_loss(CNode node, double vl_value) noexcept:
    """Apply virtual loss from node up to root."""
    while node is not None:
        node.n += 1
        node.W -= vl_value
        node.Q = node.W / <double>node.n
        node = node.parent


cdef void _undo_virtual_loss(CNode node, double vl_value) noexcept:
    """Undo virtual loss from node up to root."""
    while node is not None:
        node.n -= 1
        node.W += vl_value
        if node.n > 0:
            node.Q = node.W / <double>node.n
        else:
            node.Q = 0.0
        node = node.parent


cdef class CMCTS:
    """Cython MCTS — same interface as Python MCTS."""
    cdef public object game
    cdef public object net
    cdef public object last_root

    def __init__(self, game, net):
        self.game = game
        self.net = net
        self.last_root = None

    def get_policy(self, int num_simulations, state, bint add_dirichlet=False):
        """Run MCTS and return visit-count policy."""
        if state.terminal:
            raise ValueError("Called get_policy with terminal state")

        cdef CNode root = CNode(None, state, self.game, self.net)
        if add_dirichlet:
            root.P = _dirichlet_noise(root.P, 0.03, 0.25)

        cdef int sim
        for sim in range(num_simulations):
            self._search(root)

        # Build policy from visit counts
        cdef int action_size = len(root._avail_mask)
        pi = np.zeros(action_size, dtype=np.float64)
        cdef double[:] pi_view = pi
        cdef int i, action
        cdef CNode child

        for i in range(root._num_available):
            action = root._avail[i]
            child = <CNode>root.children[action]
            if child is not None:
                pi_view[action] = <double>child.n / <double>root.n

        self.last_root = root
        return pi

    cdef void _search(self, CNode root):
        cdef CNode selected = self._tree_policy(root)
        cdef double value = _evaluate(selected)
        _backpropagate(value, selected)

    cdef CNode _tree_policy(self, CNode node):
        cdef int best
        cdef CNode child
        while not node.is_terminal:
            best = _best_action(node)
            if node.children[best] is None:
                child = CNode(node, self.game.step(node.state, best),
                              self.game, self.net)
                node.children[best] = child
                return child
            node = <CNode>node.children[best]
        return node

    # --- Batched parallel self-play methods ---

    def search_expand(self, CNode root):
        """Select+expand phase for batched eval. Returns leaf or None."""
        cdef CNode leaf = self._tree_policy_deferred(root)
        if leaf.is_terminal or leaf.P is not None:
            _backpropagate(_evaluate(leaf), leaf)
            return None
        return leaf

    def search_backup(self, CNode node):
        """Complete evaluate + backpropagate for a resolved node."""
        _backpropagate(_evaluate(node), node)

    cdef CNode _tree_policy_deferred(self, CNode node):
        """Select+expand without neural net eval."""
        cdef int best
        cdef CNode child
        while not node.is_terminal:
            best = _best_action(node)
            if node.children[best] is None:
                child = CNode(node, self.game.step(node.state, best),
                              self.game)  # net=None → deferred
                node.children[best] = child
                return child
            node = <CNode>node.children[best]
        return node

    # --- Virtual loss methods for multi-select batching ---

    def search_expand_vl(self, CNode root, double vl_value=3.0):
        """Select+expand with virtual loss. Returns (leaf, path) or (None, None)."""
        cdef CNode node = root
        cdef CNode child
        cdef int best
        cdef list path = []

        # Tree policy with VL
        while not node.is_terminal:
            path.append(node)
            if node.P is None:
                # Node created in a previous select this round, not yet evaluated
                _apply_virtual_loss(node, vl_value)
                return node, path
            best = _best_action(node)
            if node.children[best] is None:
                child = CNode(node, self.game.step(node.state, best),
                              self.game)  # net=None → deferred
                node.children[best] = child
                path.append(child)
                _apply_virtual_loss(child, vl_value)
                return child, path
            node = <CNode>node.children[best]
        path.append(node)

        # Terminal or already evaluated — no VL needed, handle immediately
        if node.is_terminal or node.P is not None:
            _backpropagate(_evaluate(node), node)
            return None, None
        # Shouldn't reach here, but handle gracefully
        _apply_virtual_loss(node, vl_value)
        return node, path

    def search_backup_vl(self, CNode node, list path, double vl_value=3.0):
        """Undo virtual loss on path, then normal backprop."""
        _undo_virtual_loss(node, vl_value)
        _backpropagate(_evaluate(node), node)

    def undo_virtual_loss(self, CNode node, list path, double vl_value=3.0):
        """Just undo VL without backprop (cleanup/dedup)."""
        _undo_virtual_loss(node, vl_value)
