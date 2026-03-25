"""Verification tests for MCTS value conventions, backpropagation, and search.

Run: python -m tests.test_mcts
"""
import math
import numpy as np
from games.connect4 import Connect4Game as CConnect4Game, CConnect4State
from utils import load_game, print_board as _print_board
from mcts import MCTS, Node, add_dirichlet_noise
from tests.test_utils import TestCounter

game = CConnect4Game()
tc = TestCounter()
check = tc.check


def print_board(board):
    _print_board(board, "connect4")


class MockNet:
    """Mock neural network that returns fixed value and uniform policy."""
    def __init__(self, value=0.0):
        self.fixed_value = value
        self.call_count = 0
        self.last_input = None

    def predict(self, state_input):
        self.call_count += 1
        self.last_input = state_input
        policy = np.ones(7) / 7
        return self.fixed_value, policy

    def batch_predict(self, state_inputs, detailed_timing=False):
        self.call_count += len(state_inputs)
        values = [self.fixed_value] * len(state_inputs)
        policies = [np.ones(7) / 7 for _ in state_inputs]
        if detailed_timing:
            return values, policies, {
                "transfer_time": 0, "forward_time": 0, "result_time": 0
            }
        return values, policies


class PositionAwareNet:
    """Mock network that returns different values for different players.

    With relative encoding (no player indicator channels), infers player
    from piece counts: equal my/opp pieces = first mover (X) to move,
    more my pieces = second mover (O) to move.
    """
    def __init__(self, value_for_x_to_move=0.0, value_for_o_to_move=0.0):
        self.vx = value_for_x_to_move  # predicted when X (player=-1) to move
        self.vo = value_for_o_to_move  # predicted when O (player=1) to move
        self.call_count = 0

    def predict(self, state_input):
        self.call_count += 1
        # With relative encoding: ch0 = my pieces, ch1 = opponent pieces
        # X moves first, so equal piece counts = X to move
        my_count = state_input[0].sum()
        opp_count = state_input[1].sum()
        if my_count == opp_count:
            v = self.vx  # X to move (equal pieces = first mover's turn)
        else:
            v = self.vo  # O to move (opp has one more = second mover's turn)
        return v, np.ones(7) / 7

    def batch_predict(self, state_inputs, detailed_timing=False):
        self.call_count += len(state_inputs)
        values = []
        policies = []
        for si in state_inputs:
            v, p = self.predict(si)
            values.append(v)
            policies.append(p)
        if detailed_timing:
            return values, policies, {
                "transfer_time": 0, "forward_time": 0, "result_time": 0
            }
        return values, policies


# ============================================================
# 1. _evaluate() for terminal states
# ============================================================
print("\n=== _evaluate() Terminal State Tests ===")

mock = MockNet(0.0)
mcts = MCTS(game, mock)

# X wins (terminal_value = -1)
# Create a board where X has 4 horizontal in row 0
board = np.zeros((6, 7), dtype="int")
board[0][0:4] = -1  # X wins
state = CConnect4State.from_board(board, player=1)  # It would be O's turn
check("X wins: terminal=True", state.terminal)
check("X wins: terminal_value=-1", state.terminal_value == -1)

node = Node(None, state, game, mock)
val = mcts._evaluate(node)
# evaluate = -terminal_value * player = -(-1) * 1 = 1
check("X wins, O to move: evaluate returns 1",
      abs(val - 1.0) < 1e-6, f"got {val}")

# O wins (terminal_value = 1)
board = np.zeros((6, 7), dtype="int")
board[0][0:4] = 1  # O wins
state = CConnect4State.from_board(board, player=-1)  # It would be X's turn
node = Node(None, state, game, mock)
val = mcts._evaluate(node)
# evaluate = -1 * (-1) = 1
check("O wins, X to move: evaluate returns 1",
      abs(val - 1.0) < 1e-6, f"got {val}")



# ============================================================
# 2. _evaluate() for non-terminal states
# ============================================================
print("\n=== _evaluate() Non-Terminal Tests ===")

# Network predicts +0.5 (relative: "I'm winning slightly")
# Position: O to move (player=1)
mock_pos = MockNet(0.5)
mcts_pos = MCTS(game, mock_pos)
s = game.new_game()  # player=-1 (X to move)
s = game.step(s, 3)  # player=1 (O to move)

node = Node(None, s, game, mock_pos)
val = mcts_pos._evaluate(node)
# relative: evaluate = -nnet_value = -0.5
check("nnet=+0.5, O to move: evaluate = -0.5",
      abs(val - (-0.5)) < 1e-6, f"got {val}")

# Network predicts -0.5 (relative: "I'm losing slightly")
# Position: X to move (player=-1)
mock_neg = MockNet(-0.5)
mcts_neg = MCTS(game, mock_neg)
s = game.new_game()  # player=-1

node = Node(None, s, game, mock_neg)
val = mcts_neg._evaluate(node)
# relative: evaluate = -(-0.5) = +0.5
check("nnet=-0.5, X to move: evaluate = +0.5",
      abs(val - 0.5) < 1e-6, f"got {val}")


# ============================================================
# 3. Backpropagation sign conventions
# ============================================================
print("\n=== Backpropagation Convention Tests ===")

# Create a simple tree: root -> child (terminal, X wins)
mock = MockNet(0.0)
mcts = MCTS(game, mock)

s = game.new_game()  # X to move (player=-1)
root = Node(None, s, game, mock)
root.P = np.ones(7) / 7

# X plays col 0, create child that leads to X winning immediately
# For testing, create a board where X has 3 in a row and plays the 4th
board = np.zeros((6, 7), dtype="int")
board[0][0:3] = -1  # X has 3 in row 0
state_before = CConnect4State.from_board(board, player=-1)  # X to move
root = Node(None, state_before, game, mock)
root.P = np.ones(7) / 7

# X plays col 3 → wins
child_state = game.step(state_before, 3)
check("X plays col 3: terminal", child_state.terminal)
check("X plays col 3: X wins (val=-1)", child_state.terminal_value == -1)

child = Node(root, child_state, game, mock)
root.children[3] = child

val = mcts._evaluate(child)
# -(-1) * 1 = 1 (child player is 1 because X just moved)
check("Winning child evaluate = 1", abs(val - 1.0) < 1e-6, f"got {val}")

mcts._backpropagate(val, child)
check("After backprop: child.Q = 1.0", abs(child.Q - 1.0) < 1e-6, f"got {child.Q}")
check("After backprop: child.n = 1", child.n == 1)
check("After backprop: root.Q = -1.0", abs(root.Q - (-1.0)) < 1e-6, f"got {root.Q}")
check("After backprop: root.n = 1", root.n == 1)

# Verify PUCT: root selects col 3 because child.Q = 1.0 is best
best = mcts._best_action(root)
check("PUCT selects winning move (col 3)", best == 3, f"got col {best}")


# ============================================================
# 4. Q value sign: child.Q positive = good for parent
# ============================================================
print("\n=== Q Sign Convention Tests ===")

# Scenario: root is X to move, network predicts "I'm losing" (nnet=-1 relative)
mock_xwins = MockNet(-1.0)
mcts_xw = MCTS(game, mock_xwins)

s = game.new_game()  # X to move
root = Node(None, s, game, mock_xwins)
root.P = np.ones(7) / 7

# Run one MCTS simulation
mcts_xw._search(root)

# The child that was expanded should have:
# - nnet_value = -1.0 (network says "I'm losing")
# - relative: _evaluate(child) = -(-1.0) = 1.0
# - child.Q = 1.0 (from first backprop)
# - root.Q = -1.0

# Find the expanded child
expanded = None
for a in root.available_actions:
    if root.children[a] is not None:
        expanded = root.children[a]
        expanded_action = a
        break

check("One child expanded", expanded is not None)
if expanded:
    check(f"Child Q = +1.0 (good for parent X, X wins)",
          abs(expanded.Q - 1.0) < 1e-6, f"got {expanded.Q}")
    check(f"Root Q = -1.0 (negated from child)",
          abs(root.Q - (-1.0)) < 1e-6, f"got {root.Q}")

# Scenario: root is O to move, network predicts "I'm losing" (nnet=-1 relative)
mock_xloss = MockNet(-1.0)
mcts_xl = MCTS(game, mock_xloss)

s = game.new_game()
s = game.step(s, 3)  # O to move
root = Node(None, s, game, mock_xloss)
root.P = np.ones(7) / 7

mcts_xl._search(root)

expanded = None
for a in root.available_actions:
    if root.children[a] is not None:
        expanded = root.children[a]
        break

if expanded:
    # child.state.player = -1 (X to move after O plays)
    # relative: _evaluate = -(-1.0) = +1.0
    # child.Q = +1.0
    # root.Q = -1.0
    check("When O to move, nnet=-1 relative: child.Q = +1.0",
          abs(expanded.Q - 1.0) < 1e-6, f"got {expanded.Q}")
    check("When O to move, nnet=-1 relative: root.Q = -1.0",
          abs(root.Q - (-1.0)) < 1e-6, f"got {root.Q}")


# ============================================================
# 5. PUCT selects winning move in trivial position
# ============================================================
print("\n=== PUCT Winning Move Selection ===")

# Position where X can win by playing col 3 (needs 4th in row)
board = np.zeros((6, 7), dtype="int")
board[0][0:3] = -1  # X: row 0, cols 0-2
# Add some O pieces so the position is realistic
board[0][4] = 1
board[0][5] = 1

state = CConnect4State.from_board(board, player=-1)  # X to move
check("Pre-win position is NOT terminal", not state.terminal)

# Use a network that's neutral (doesn't bias the search)
mock_neutral = MockNet(0.0)
mcts_n = MCTS(game, mock_neutral)
pi = mcts_n.get_policy(50, state)
root = mcts_n.last_root

best = np.argmax(pi)
print(f"  MCTS with 50 sims (neutral net):")
print(f"  Best action: col {best}")
for a in range(7):
    child = root.children[a]
    if child is not None:
        marker = " <-- WIN" if a == 3 else ""
        print(f"    Col {a}: N={child.n:>3} Q={child.Q:+.3f}{marker}")

# After 50 sims, MCTS should find that col 3 wins (terminal child with Q=1)
col3_child = root.children[3]
if col3_child is not None:
    check("Col 3 child has Q ~= 1.0 (winning for X)",
          col3_child.Q > 0.5, f"Q={col3_child.Q}")
    check("Col 3 gets most visits", pi[3] > pi.max() * 0.5)
else:
    check("Col 3 was explored", False, "col 3 child is None")


# ============================================================
# 6. PUCT blocks opponent winning move
# ============================================================
print("\n=== PUCT Blocking Move ===")

# O has 3 in a row, X must block at col 3
board = np.zeros((6, 7), dtype="int")
board[0][0:3] = 1   # O: row 0, cols 0-2 (3 in a row)
board[1][0] = -1     # Some X pieces
board[1][1] = -1

state = CConnect4State.from_board(board, player=-1)  # X to move
check("Blocking position is NOT terminal", not state.terminal)

# Verify O would win at col 3
o_wins = game.step(CConnect4State.from_board(board, player=1), 3)
check("O would win at col 3", o_wins.terminal and o_wins.terminal_value == 1)

mock_neutral = MockNet(0.0)
mcts_b = MCTS(game, mock_neutral)
pi = mcts_b.get_policy(100, state)
root = mcts_b.last_root

best = np.argmax(pi)
print(f"  MCTS with 100 sims (neutral net), X must block col 3:")
for a in range(7):
    child = root.children[a]
    if child is not None:
        marker = ""
        if a == 3:
            marker = " <-- BLOCK"
        if a == best:
            marker += " <-- BEST"
        print(f"    Col {a}: N={child.n:>3} Q={child.Q:+.3f}{marker}")

# X should play col 3 to block O's win
check("X blocks at col 3", best == 3, f"best was col {best}")


# ============================================================
# 7. Virtual loss apply/undo symmetry
# ============================================================
print("\n=== Virtual Loss Symmetry ===")

mock = MockNet(0.0)
mcts_vl = MCTS(game, mock)

s = game.new_game()
root = Node(None, s, game, mock)
root.P = np.ones(7) / 7

# Run a simulation to create a child
mcts_vl._search(root)
original_n = root.n
original_Q = root.Q
original_W = root.W

# Apply virtual loss
mcts_vl._apply_virtual_loss(root, 3.0)
check("After VL apply: n increased", root.n == original_n + 1)
check("After VL apply: W decreased", root.W == original_W - 3.0)

# Undo virtual loss
mcts_vl._undo_virtual_loss(root, 3.0)
check("After VL undo: n restored", root.n == original_n)
check("After VL undo: W restored", abs(root.W - original_W) < 1e-6)
check("After VL undo: Q restored", abs(root.Q - original_Q) < 1e-6)


# ============================================================
# 8. Deferred evaluation (search_expand + search_backup)
# ============================================================
print("\n=== Deferred Evaluation Tests ===")

mock = MockNet(0.0)
mcts_d = MCTS(game, mock)

s = game.new_game()
root = Node(None, s, game, mock)
root.P = np.ones(7) / 7

# search_expand should return an unevaluated leaf
leaf = mcts_d.search_expand(root)
if leaf is not None:
    check("Deferred leaf has P=None before resolve", leaf.P is None)

    # Resolve with mock values
    leaf.resolve(0.0, np.ones(7) / 7)
    check("After resolve: nnet_value = 0.0", leaf.nnet_value == 0.0)

    # Backup
    mcts_d.search_backup(leaf)
    check("After backup: root.n = 1", root.n == 1)
else:
    check("search_expand returned a leaf", False, "returned None")


# ============================================================
# 9. Training target consistency check
# ============================================================
print("\n=== Training Target Consistency ===")
print("  Verifying: nnet_value convention matches training targets")

# With relative encoding, perfect network predicts:
#   +1 when current player will win ("I'm winning")
#   -1 when current player will lose ("I'm losing")
# _evaluate = -nnet_value (for parent's perspective)

# Scenario 1: X will win. Perfect network: X sees +1, O sees -1.
perfect_net = PositionAwareNet(value_for_x_to_move=1.0, value_for_o_to_move=-1.0)
mcts_p = MCTS(game, perfect_net)
s = game.new_game()  # X to move
root = Node(None, s, game, perfect_net)
val = mcts_p._evaluate(root)
# relative: -(+1.0) = -1.0
check("Perfect net, X wins, X to move: evaluate = -1.0",
      abs(val - (-1.0)) < 1e-6, f"got {val}")
print(f"    Interpretation: root.Q will be -1.0")
print(f"    This means: from root's parent's perspective (hypothetical), this is bad")
print(f"    But root IS the search root, so we care about child.Q, not root.Q")

# Now run a full search: root (X to move) → children (O to move, nnet=-1)
# child evaluate = -(-1.0) = 1.0
# child.Q = 1.0 → good for parent X → X explores further
mcts_p._search(root)
for a in root.available_actions:
    child = root.children[a]
    if child is not None:
        check(f"Perfect net X wins: child.Q = +1.0 (col {a})",
              abs(child.Q - 1.0) < 1e-6, f"got {child.Q}")
        break

# Scenario 2: O will win. Perfect network: X sees -1, O sees +1.
perfect_net2 = PositionAwareNet(value_for_x_to_move=-1.0, value_for_o_to_move=1.0)
mcts_p2 = MCTS(game, perfect_net2)
s = game.new_game()  # X to move
root = Node(None, s, game, perfect_net2)
root.P = np.ones(7) / 7

mcts_p2._search(root)
for a in root.available_actions:
    child = root.children[a]
    if child is not None:
        # child (O to move), nnet=+1 (O winning, relative)
        # evaluate = -(+1) = -1
        # child.Q = -1 → bad for parent X (because O wins)
        check(f"Perfect net O wins: child.Q = -1.0 (col {a})",
              abs(child.Q - (-1.0)) < 1e-6, f"got {child.Q}")
        break


# ============================================================
# 10. Multi-simulation convergence
# ============================================================
print("\n=== Multi-Simulation Convergence ===")

# With a neutral network and many simulations, MCTS should
# distribute visits roughly proportional to the prior (uniform → equal)
mock = MockNet(0.0)
mcts_conv = MCTS(game, mock)
s = game.new_game()
pi = mcts_conv.get_policy(700, s)  # 100 per action
root = mcts_conv.last_root

print(f"  700 sims, neutral network, new game:")
visit_counts = []
for a in range(7):
    child = root.children[a]
    n = child.n if child else 0
    q = f"{child.Q:+.3f}" if child else "  -  "
    print(f"    Col {a}: N={n:>4} Q={q} pi={pi[a]:.3f}")
    visit_counts.append(n)

# With uniform prior and zero value, visits should be roughly equal
min_v = min(visit_counts)
max_v = max(visit_counts)
ratio = max_v / max(min_v, 1)
check(f"Uniform visits ratio (max/min = {ratio:.1f})",
      ratio < 3.0, f"max={max_v} min={min_v}")


# ============================================================
# 11. Dirichlet noise
# ============================================================
print("\n=== Dirichlet Noise ===")

uniform = np.ones(7) / 7
noisy = add_dirichlet_noise(uniform, 0.03, 0.25)
check("Noisy policy sums to ~1.0", abs(noisy.sum() - 1.0) < 1e-6)
check("Noisy policy has 7 elements", len(noisy) == 7)
check("Noisy policy all positive", all(noisy > 0))
check("Noisy policy differs from uniform", not np.allclose(noisy, uniform, atol=0.01))


# ============================================================
# Summary
# ============================================================
tc.summary("MCTS logic")
