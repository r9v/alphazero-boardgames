"""Tests for Santorini D4 symmetry augmentation.

Run: python -m tests.test_santorini_symmetry
"""
import numpy as np
from games.santorini.game import SantoriniGame, GameState, DIRECTIONS
from games.santorini.symmetry import (
    get_symmetries, _transform_state, _transform_pos, _transform_policy,
    _needs_worker_swap, DIR_PERMS, _ACTION_REMAP, BOARD_SIZE,
)

game = SantoriniGame()
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


# ----------------------------------------------------------------
# 1. Basic sanity
# ----------------------------------------------------------------
print("\n=== 1. Basic sanity ===")

state = game.new_game()
inp = game.state_to_input(state)
policy = np.zeros(128, dtype=np.float32)
# Set some valid actions with nonzero prob
valid = np.where(state.available_actions == 1)[0]
for a in valid[:5]:
    policy[a] = 1.0
policy /= policy.sum()

syms = get_symmetries(inp, policy)

check("get_symmetries returns 8 elements", len(syms) == 8)
check("first element is identity (state)",
      np.array_equal(syms[0][0], inp))
check("first element is identity (policy)",
      np.array_equal(syms[0][1], policy))

# All states should have same shape
for i, (s, p) in enumerate(syms):
    check(f"sym[{i}] state shape", s.shape == inp.shape,
          f"got {s.shape}")
    check(f"sym[{i}] policy shape", p.shape == policy.shape,
          f"got {p.shape}")

# ----------------------------------------------------------------
# 2. Policy probability mass is preserved
# ----------------------------------------------------------------
print("\n=== 2. Policy mass preservation ===")

for i, (s, p) in enumerate(syms):
    check(f"sym[{i}] policy sum ~= 1.0",
          abs(p.sum() - 1.0) < 1e-6,
          f"got {p.sum()}")
    check(f"sym[{i}] no negative probs",
          (p >= 0).all())

# ----------------------------------------------------------------
# 3. Spatial transform group properties
# ----------------------------------------------------------------
print("\n=== 3. Group properties ===")

# rot90 x 4 = identity
s = inp.copy()
for _ in range(4):
    s = _transform_state(s, 1)  # rot90 CW
check("rot90 x 4 = identity", np.allclose(s, inp))

# rot180 x 2 = identity
s = _transform_state(_transform_state(inp, 2), 2)
check("rot180 x 2 = identity", np.allclose(s, inp))

# flip_h x 2 = identity
s = _transform_state(_transform_state(inp, 4), 4)
check("flip_h x 2 = identity", np.allclose(s, inp))

# flip_v x 2 = identity
s = _transform_state(_transform_state(inp, 5), 5)
check("flip_v x 2 = identity", np.allclose(s, inp))

# flip_diag x 2 = identity
s = _transform_state(_transform_state(inp, 6), 6)
check("flip_diag x 2 = identity", np.allclose(s, inp))

# flip_anti x 2 = identity
s = _transform_state(_transform_state(inp, 7), 7)
check("flip_anti x 2 = identity", np.allclose(s, inp))

# rot90 CW = flip_diag then flip_h
s1 = _transform_state(inp, 1)
s2 = _transform_state(_transform_state(inp, 6), 4)
check("rot90 CW = flip_diag o flip_h", np.allclose(s1, s2))

# ----------------------------------------------------------------
# 4. Position transform correctness
# ----------------------------------------------------------------
print("\n=== 4. Position transforms ===")

# rot90 CW: (0,0) -> (0,4), (0,4) -> (4,4), (4,4) -> (4,0), (4,0) -> (0,0)
check("rot90 CW (0,0)->(0,4)",
      _transform_pos(0, 0, 1) == (0, 4))
check("rot90 CW (0,4)->(4,4)",
      _transform_pos(0, 4, 1) == (4, 4))
check("rot90 CW (4,4)->(4,0)",
      _transform_pos(4, 4, 1) == (4, 0))
check("rot90 CW (4,0)->(0,0)",
      _transform_pos(4, 0, 1) == (0, 0))

# rot180: (0,0) -> (4,4)
check("rot180 (0,0)->(4,4)",
      _transform_pos(0, 0, 2) == (4, 4))
check("rot180 (2,2)->(2,2)",
      _transform_pos(2, 2, 2) == (2, 2),
      f"got {_transform_pos(2, 2, 2)}")

# flip_h: (r,c) -> (r, 4-c)
check("flip_h (1,3)->(1,1)",
      _transform_pos(1, 3, 4) == (1, 1))

# flip_v: (r,c) -> (4-r, c)
check("flip_v (1,3)->(3,3)",
      _transform_pos(1, 3, 5) == (3, 3))

# flip_diag (transpose): (r,c) -> (c,r)
check("flip_diag (1,3)->(3,1)",
      _transform_pos(1, 3, 6) == (3, 1))

# flip_anti: (r,c) -> (4-c, 4-r)
check("flip_anti (1,3)->(1,3)",
      _transform_pos(1, 3, 7) == (1, 3))
check("flip_anti (0,0)->(4,4)",
      _transform_pos(0, 0, 7) == (4, 4))

# ----------------------------------------------------------------
# 5. Position transform matches state transform
# ----------------------------------------------------------------
print("\n=== 5. Position vs State consistency ===")

# Place a marker at (1, 3) on channel 0, transform, verify it moves correctly
for sym_idx in range(8):
    test_state = np.zeros((1, 5, 5), dtype=np.float32)
    test_state[0, 1, 3] = 1.0
    transformed = _transform_state(test_state, sym_idx)
    expected_r, expected_c = _transform_pos(1, 3, sym_idx)
    actual_pos = list(zip(*np.where(transformed[0] > 0.5)))
    check(f"sym[{sym_idx}] posvsstate match for (1,3)",
          len(actual_pos) == 1 and actual_pos[0] == (expected_r, expected_c),
          f"expected ({expected_r},{expected_c}), got {actual_pos}")

# ----------------------------------------------------------------
# 6. Direction permutation correctness
# ----------------------------------------------------------------
print("\n=== 6. Direction permutations ===")

# Verify each direction permutation by checking that transformed direction
# vectors match the spatial transform applied to the original direction vectors.
for sym_idx in range(8):
    dp = DIR_PERMS[sym_idx]
    all_ok = True
    for d_idx, (dr, dc) in enumerate(DIRECTIONS):
        # Original direction from center (2,2) reaches (2+dr, 2+dc)
        orig_dest = (2 + dr, 2 + dc)
        # Transform both center and destination
        t_center = _transform_pos(2, 2, sym_idx)
        t_dest = _transform_pos(orig_dest[0], orig_dest[1], sym_idx)
        # New direction vector
        new_dr = t_dest[0] - t_center[0]
        new_dc = t_dest[1] - t_center[1]
        # Find which direction index this corresponds to
        new_d_idx = DIRECTIONS.index((new_dr, new_dc))
        if dp[d_idx] != new_d_idx:
            all_ok = False
            check(f"sym[{sym_idx}] dir[{d_idx}]", False,
                  f"expected dir {new_d_idx}, table says {dp[d_idx]}")
    if all_ok:
        check(f"sym[{sym_idx}] all 8 directions correct", True)

# ----------------------------------------------------------------
# 7. Worker swap detection
# ----------------------------------------------------------------
print("\n=== 7. Worker swap detection ===")

# Build a state where workers are at (1,0) and (3,4)
# Sorted: (1,0) is w0, (3,4) is w1
test_state = np.zeros((7, 5, 5), dtype=np.float32)
test_state[0] = 1.0  # all level 0
test_state[5, 1, 0] = 1.0  # current player worker at (1,0)
test_state[5, 3, 4] = 1.0  # current player worker at (3,4)
test_state[6, 0, 2] = 1.0  # opponent worker
test_state[6, 4, 2] = 1.0  # opponent worker

for sym_idx in range(8):
    t0 = _transform_pos(1, 0, sym_idx)
    t1 = _transform_pos(3, 4, sym_idx)
    expected_swap = t0 > t1  # sort order flipped
    actual_swap = _needs_worker_swap(test_state, sym_idx)
    check(f"sym[{sym_idx}] worker swap: {actual_swap} (w0->{t0}, w1->{t1})",
          actual_swap == expected_swap,
          f"expected {expected_swap}")

# ----------------------------------------------------------------
# 8. Full roundtrip: transform then inverse = identity
# ----------------------------------------------------------------
print("\n=== 8. Roundtrip (transform + inverse) ===")

# Inverse table: for each symmetry, which symmetry undoes it
# identity->identity, rot90->rot270, rot180->rot180,
# rot270->rot90, flip_*->flip_* (self-inverse)
INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

# Use a non-trivial state
state = game.new_game()
# Make a few moves to create an interesting board
actions = []
s = state
for _ in range(6):
    valid = np.where(s.available_actions == 1)[0]
    a = valid[len(valid) // 2]  # pick middle action
    actions.append(a)
    s = game.step(s, a)

inp = game.state_to_input(s)
policy = np.random.dirichlet(np.ones(128)).astype(np.float32)
# Zero out invalid actions and renormalize
policy[s.available_actions == 0] = 0
if policy.sum() > 0:
    policy /= policy.sum()

for sym_idx in range(8):
    inv_idx = INVERSE[sym_idx]
    # Forward transform
    s1 = _transform_state(inp, sym_idx)
    swap1 = _needs_worker_swap(inp, sym_idx)
    p1 = _transform_policy(policy, sym_idx, swap1)
    # Inverse transform
    s2 = _transform_state(s1, inv_idx)
    swap2 = _needs_worker_swap(s1, inv_idx)
    p2 = _transform_policy(p1, inv_idx, swap2)

    check(f"sym[{sym_idx}] roundtrip state",
          np.allclose(s2, inp),
          f"max diff={np.abs(s2 - inp).max()}")
    check(f"sym[{sym_idx}] roundtrip policy",
          np.allclose(p2, policy, atol=1e-7),
          f"max diff={np.abs(p2 - policy).max()}")

# ----------------------------------------------------------------
# 9. All 8 states are distinct (for non-symmetric positions)
# ----------------------------------------------------------------
print("\n=== 9. Distinct transforms ===")

# Use the non-trivial state from above (asymmetric board)
syms = get_symmetries(inp, policy)
states_list = [s.tobytes() for s, p in syms]
n_unique = len(set(states_list))
check(f"all 8 transforms produce distinct states (got {n_unique})",
      n_unique == 8,
      "some transforms produced identical states on asymmetric board")

# ----------------------------------------------------------------
# 10. Game interface integration
# ----------------------------------------------------------------
print("\n=== 10. Game interface ===")

state = game.new_game()
inp = game.state_to_input(state)
valid = np.where(state.available_actions == 1)[0]
policy = np.zeros(128, dtype=np.float32)
for a in valid:
    policy[a] = 1.0 / len(valid)

syms = game.get_symmetries(inp, policy)
check("game.get_symmetries returns 8 elements", len(syms) == 8)
check("game.get_symmetries first is identity",
      np.array_equal(syms[0][0], inp) and np.array_equal(syms[0][1], policy))

# ----------------------------------------------------------------
# 11. Action remapping preserves valid actions
# ----------------------------------------------------------------
print("\n=== 11. Action validity preservation ===")

# For each symmetry, if we transform the state AND transform which actions
# are valid, the number of valid actions should be preserved
state = game.new_game()
inp = game.state_to_input(state)
n_valid_orig = state.available_actions.sum()

# Make policy = available_actions (uniform over valid)
policy = state.available_actions.astype(np.float32)
if policy.sum() > 0:
    policy /= policy.sum()

syms = get_symmetries(inp, policy)
for i, (s, p) in enumerate(syms):
    n_valid_sym = (p > 0).sum()
    check(f"sym[{i}] preserves valid action count ({n_valid_orig})",
          n_valid_sym == n_valid_orig,
          f"got {n_valid_sym}")

# ----------------------------------------------------------------
# 12. Verify with multiple game states
# ----------------------------------------------------------------
print("\n=== 12. Multi-state roundtrip ===")

# Play random games and verify roundtrip at each position
np.random.seed(42)
n_tested = 0
n_errors = 0
s = game.new_game()
for move in range(20):
    if s.terminal:
        break
    valid = np.where(s.available_actions == 1)[0]
    a = np.random.choice(valid)
    s = game.step(s, a)
    if s.terminal:
        break

    inp = game.state_to_input(s)
    policy = np.random.dirichlet(np.ones(128)).astype(np.float32)
    policy[s.available_actions == 0] = 0
    if policy.sum() > 0:
        policy /= policy.sum()

    for sym_idx in range(8):
        inv_idx = INVERSE[sym_idx]
        s1 = _transform_state(inp, sym_idx)
        swap1 = _needs_worker_swap(inp, sym_idx)
        p1 = _transform_policy(policy, sym_idx, swap1)
        s2 = _transform_state(s1, inv_idx)
        swap2 = _needs_worker_swap(s1, inv_idx)
        p2 = _transform_policy(p1, inv_idx, swap2)
        n_tested += 1
        if not np.allclose(s2, inp) or not np.allclose(p2, policy, atol=1e-7):
            n_errors += 1

check(f"multi-state roundtrip ({n_tested} tests, {n_errors} errors)",
      n_errors == 0)


# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("All tests passed!")
else:
    print("SOME TESTS FAILED")
