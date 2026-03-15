"""D4 symmetry augmentation for Santorini 5x5 board.

The D4 group has 8 elements (4 rotations + 4 reflections) that preserve
the square grid. For each symmetry, we transform:
  1. The state tensor (C, 5, 5) — spatial transform
  2. The policy vector (128,) — remap action indices

Action encoding: action = worker_idx * 64 + move_dir * 8 + build_dir
  - worker_idx: 0 or 1 (workers sorted by (row, col))
  - move_dir / build_dir: index into DIRECTIONS
  - DIRECTIONS: [N, NE, E, SE, S, SW, W, NW] = indices 0-7

Direction permutation: each D4 element permutes the 8 compass directions.
Worker swap: after spatial transform, the (row, col) sort order of the two
workers may change, requiring worker_idx 0<->1 swap.
"""
import numpy as np

BOARD_SIZE = 5

# Direction permutation tables for each D4 element.
# DIRECTIONS = [N, NE, E, SE, S, SW, W, NW] = indices 0-7
# Each entry maps old_dir_index -> new_dir_index
#
# For compass directions on a grid, rotation by 90° CW shifts each
# direction by +2 (mod 8). Reflections reverse/remap directions.
DIR_PERMS = [
    [0, 1, 2, 3, 4, 5, 6, 7],  # 0: identity
    [2, 3, 4, 5, 6, 7, 0, 1],  # 1: rot90 CW  (d -> d+2 mod 8)
    [4, 5, 6, 7, 0, 1, 2, 3],  # 2: rot180    (d -> d+4 mod 8)
    [6, 7, 0, 1, 2, 3, 4, 5],  # 3: rot270 CW (d -> d+6 mod 8)
    [0, 7, 6, 5, 4, 3, 2, 1],  # 4: flip horizontal (reflect cols)
    [4, 3, 2, 1, 0, 7, 6, 5],  # 5: flip vertical (reflect rows)
    [6, 5, 4, 3, 2, 1, 0, 7],  # 6: flip main diagonal (transpose)
    [2, 1, 0, 7, 6, 5, 4, 3],  # 7: flip anti-diagonal
]

# Precompute action remapping tables for each symmetry (without worker swap).
# _ACTION_REMAP[sym][old_action] = new_action (assuming no worker swap)
_ACTION_REMAP = np.zeros((8, 128), dtype=np.int32)
for _sym in range(8):
    _dp = DIR_PERMS[_sym]
    for _w in range(2):
        for _md in range(8):
            for _bd in range(8):
                _old = _w * 64 + _md * 8 + _bd
                _new = _w * 64 + _dp[_md] * 8 + _dp[_bd]
                _ACTION_REMAP[_sym][_old] = _new


def _transform_state(state, sym_idx):
    """Apply spatial transform to state tensor (C, H, W)."""
    if sym_idx == 0:
        return state  # identity
    elif sym_idx == 1:  # rot90 CW
        return np.rot90(state, k=-1, axes=(1, 2)).copy()
    elif sym_idx == 2:  # rot180
        return np.rot90(state, k=2, axes=(1, 2)).copy()
    elif sym_idx == 3:  # rot270 CW
        return np.rot90(state, k=1, axes=(1, 2)).copy()
    elif sym_idx == 4:  # flip horizontal (reflect columns)
        return np.ascontiguousarray(state[:, :, ::-1])
    elif sym_idx == 5:  # flip vertical (reflect rows)
        return np.ascontiguousarray(state[:, ::-1, :])
    elif sym_idx == 6:  # flip main diagonal (transpose)
        return np.ascontiguousarray(state.transpose(0, 2, 1))
    elif sym_idx == 7:  # flip anti-diagonal
        return np.ascontiguousarray(state[:, ::-1, ::-1].transpose(0, 2, 1))


def _needs_worker_swap(state, sym_idx):
    """Check if worker sort order changes after spatial transform.

    Workers are indexed by sorting their (row, col) positions. After a
    spatial transform, their positions change and the sort order may flip.

    state has channels 5 (current player workers) and 6 (opponent workers).
    We only need to check channel 5 (current player) since the action
    encoding only references the current player's workers.
    """
    # Find current player's worker positions from channel 5
    positions = list(zip(*np.where(state[5] > 0.5)))
    if len(positions) != 2:
        return False

    # Workers are sorted by (r, c) in original state
    # After transform, check if the transformed positions reverse order
    p0, p1 = sorted(positions)  # original sorted order

    # Transform both positions
    t0 = _transform_pos(p0[0], p0[1], sym_idx)
    t1 = _transform_pos(p1[0], p1[1], sym_idx)

    # If transformed positions sort differently, we need to swap
    return t0 > t1


def _transform_pos(r, c, sym_idx):
    """Transform a single (r, c) position. Returns (new_r, new_c)."""
    n = BOARD_SIZE - 1  # 4 for 5x5 board
    if sym_idx == 0:
        return (r, c)
    elif sym_idx == 1:  # rot90 CW
        return (c, n - r)
    elif sym_idx == 2:  # rot180
        return (n - r, n - c)
    elif sym_idx == 3:  # rot270 CW
        return (n - c, r)
    elif sym_idx == 4:  # flip horizontal
        return (r, n - c)
    elif sym_idx == 5:  # flip vertical
        return (n - r, c)
    elif sym_idx == 6:  # flip main diagonal (transpose)
        return (c, r)
    elif sym_idx == 7:  # flip anti-diagonal
        return (n - c, n - r)


def _transform_policy(policy, sym_idx, swap_workers):
    """Remap policy vector for a given symmetry + optional worker swap."""
    remap = _ACTION_REMAP[sym_idx]
    new_policy = np.zeros(128, dtype=policy.dtype)

    for old_action in range(128):
        if policy[old_action] == 0:
            continue
        new_action = remap[old_action]
        if swap_workers:
            # Flip worker bit: action XOR 64
            new_action ^= 64
        new_policy[new_action] = policy[old_action]

    return new_policy


def _is_placement(state_input):
    """Detect placement phase from state tensor.

    During placement, channels 5+6 have fewer than 4 total workers.
    """
    n_workers = int((state_input[5] > 0.5).sum() + (state_input[6] > 0.5).sum())
    return n_workers < 4


def _transform_placement_policy(policy, sym_idx):
    """Remap placement policy (actions 0-24 = r*5+c) via position transform."""
    new_policy = np.zeros(128, dtype=policy.dtype)
    for action in range(25):
        if policy[action] == 0:
            continue
        r, c = action // 5, action % 5
        new_r, new_c = _transform_pos(r, c, sym_idx)
        new_action = new_r * 5 + new_c
        new_policy[new_action] = policy[action]
    return new_policy


def get_symmetries(state_input, policy, ownership=None):
    """Return all 8 D4 symmetries of (state_input, policy, ownership).

    Args:
        state_input: numpy array (C, 5, 5) — game state encoding
        policy: numpy array (128,) — action probability distribution
        ownership: optional numpy array (5, 5) — ownership target

    Returns:
        List of 8 (transformed_state, transformed_policy, transformed_ownership) tuples.
        First element is always the identity (original).

    Handles both placement phase (actions 0-24 = cell positions) and
    normal play (actions 0-127 = worker*64 + move_dir*8 + build_dir).
    """
    placement = _is_placement(state_input)
    result = []
    for sym_idx in range(8):
        new_state = _transform_state(state_input, sym_idx)
        if placement:
            new_policy = _transform_placement_policy(policy, sym_idx)
        else:
            swap = _needs_worker_swap(state_input, sym_idx)
            new_policy = _transform_policy(policy, sym_idx, swap)
        new_own = _transform_state(ownership[np.newaxis], sym_idx)[0] if ownership is not None else None
        result.append((new_state, new_policy, new_own))
    return result
