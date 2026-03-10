# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated Santorini game logic.

Replaces Python GameState/SantoriniGame with C-typed operations:
- Board stored as int[5][5] C array (no numpy overhead)
- Workers stored as int arrays (no dict/deepcopy)
- Available actions computed with int[5][5] occupied grid (no Python sets)
"""
import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy, memset

cnp.import_array()

DEF BOARD_SIZE = 5
DEF ACTION_SIZE = 128
DEF NUM_WORKERS = 2
DEF NUM_DIRS = 8

# 8 compass directions: N, NE, E, SE, S, SW, W, NW
cdef int DIR_R[8]
cdef int DIR_C[8]
DIR_R[0] = -1; DIR_C[0] =  0  # N
DIR_R[1] = -1; DIR_C[1] =  1  # NE
DIR_R[2] =  0; DIR_C[2] =  1  # E
DIR_R[3] =  1; DIR_C[3] =  1  # SE
DIR_R[4] =  1; DIR_C[4] =  0  # S
DIR_R[5] =  1; DIR_C[5] = -1  # SW
DIR_R[6] =  0; DIR_C[6] = -1  # W
DIR_R[7] = -1; DIR_C[7] = -1  # NW


cdef inline bint _in_bounds(int r, int c) noexcept nogil:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


cdef class CSantoriniState:
    """Santorini game state with C-typed internals."""
    cdef int _board[5][5]
    # Workers: [player_index][worker_index] where player_index 0=-1, 1=+1
    cdef int _wr[2][2]  # worker rows
    cdef int _wc[2][2]  # worker cols
    cdef public int player
    cdef public bint terminal
    cdef public int terminal_value
    cdef public bint last_turn_skipped
    cdef int _actions[128]
    cdef public CSantoriniState prev_state

    # Cached numpy views (created on demand)
    cdef object _board_np
    cdef object _actions_np

    def __init__(self):
        # Use _init_* classmethods or factory functions instead
        pass

    cdef object _make_board_np(self):
        arr = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=np.intc)
        cdef int[:, :] view = arr
        cdef int r, c
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                view[r, c] = self._board[r][c]
        return arr

    cdef object _make_actions_np(self):
        arr = np.empty(ACTION_SIZE, dtype=np.intc)
        cdef int[:] view = arr
        cdef int i
        for i in range(ACTION_SIZE):
            view[i] = self._actions[i]
        return arr

    @property
    def board(self):
        """Return board as numpy array (creates view on first access)."""
        if self._board_np is None:
            self._board_np = self._make_board_np()
        return self._board_np

    @property
    def available_actions(self):
        """Return available actions mask as numpy array."""
        if self._actions_np is None:
            self._actions_np = self._make_actions_np()
        return self._actions_np

    @property
    def workers(self):
        """Return workers as dict for GUI compatibility."""
        return {
            -1: [(self._wr[0][0], self._wc[0][0]),
                 (self._wr[0][1], self._wc[0][1])],
             1: [(self._wr[1][0], self._wc[1][0]),
                 (self._wr[1][1], self._wc[1][1])],
        }

    def _sorted_workers(self, int player):
        """Return workers sorted by (row, col) for consistent indexing."""
        cdef int pi = 0 if player == -1 else 1
        w0 = (self._wr[pi][0], self._wc[pi][0])
        w1 = (self._wr[pi][1], self._wc[pi][1])
        if w0 <= w1:
            return [w0, w1]
        return [w1, w0]


cdef int _player_idx(int player) noexcept nogil:
    """Convert player (-1 or 1) to index (0 or 1)."""
    if player == -1:
        return 0
    return 1


cdef void _sort_workers(int *wr, int *wc) noexcept nogil:
    """Sort two workers by (row, col) for consistent indexing."""
    cdef int r0 = wr[0], c0 = wc[0], r1 = wr[1], c1 = wc[1]
    if r0 > r1 or (r0 == r1 and c0 > c1):
        wr[0] = r1; wc[0] = c1
        wr[1] = r0; wc[1] = c0


cdef void _compute_available(CSantoriniState state) noexcept:
    """Compute available actions mask using C arrays."""
    cdef int occupied[5][5]
    cdef int r, c, w_idx, m_dir, b_dir
    cdef int wr, wc_val, mr, mc, br, bc
    cdef int current_level, target_level
    cdef int pi = _player_idx(state.player)
    cdef int oi = 1 - pi  # opponent index
    cdef int sorted_wr[2]
    cdef int sorted_wc[2]

    # Clear actions
    memset(state._actions, 0, ACTION_SIZE * sizeof(int))

    # Build occupied grid
    memset(occupied, 0, 25 * sizeof(int))
    occupied[state._wr[0][0]][state._wc[0][0]] = 1
    occupied[state._wr[0][1]][state._wc[0][1]] = 1
    occupied[state._wr[1][0]][state._wc[1][0]] = 1
    occupied[state._wr[1][1]][state._wc[1][1]] = 1

    # Get sorted workers for current player
    sorted_wr[0] = state._wr[pi][0]; sorted_wc[0] = state._wc[pi][0]
    sorted_wr[1] = state._wr[pi][1]; sorted_wc[1] = state._wc[pi][1]
    _sort_workers(sorted_wr, sorted_wc)

    for w_idx in range(NUM_WORKERS):
        wr = sorted_wr[w_idx]
        wc_val = sorted_wc[w_idx]
        current_level = state._board[wr][wc_val]

        for m_dir in range(NUM_DIRS):
            mr = wr + DIR_R[m_dir]
            mc = wc_val + DIR_C[m_dir]

            # Validate move
            if not _in_bounds(mr, mc):
                continue
            if occupied[mr][mc]:
                continue
            if state._board[mr][mc] >= 4:  # dome
                continue
            if state._board[mr][mc] - current_level > 1:  # too high
                continue

            # Temporarily update occupied for build check
            occupied[wr][wc_val] = 0
            occupied[mr][mc] = 1

            for b_dir in range(NUM_DIRS):
                br = mr + DIR_R[b_dir]
                bc = mc + DIR_C[b_dir]

                if not _in_bounds(br, bc):
                    continue
                if occupied[br][bc]:
                    continue
                if state._board[br][bc] >= 4:
                    continue

                state._actions[w_idx * 64 + m_dir * 8 + b_dir] = 1

            # Restore occupied
            occupied[wr][wc_val] = 1
            occupied[mr][mc] = 0


cdef CSantoriniState _new_game():
    """Create initial game state."""
    cdef CSantoriniState s = CSantoriniState.__new__(CSantoriniState)
    memset(s._board, 0, 25 * sizeof(int))
    # Player -1 workers (index 0): (0,1) and (0,3)
    s._wr[0][0] = 0; s._wc[0][0] = 1
    s._wr[0][1] = 0; s._wc[0][1] = 3
    # Player +1 workers (index 1): (4,1) and (4,3)
    s._wr[1][0] = 4; s._wc[1][0] = 1
    s._wr[1][1] = 4; s._wc[1][1] = 3
    s.player = -1
    s.last_turn_skipped = False
    s.prev_state = None
    s._board_np = None
    s._actions_np = None

    _compute_available(s)
    # Check terminal
    cdef int total = 0
    cdef int i
    for i in range(ACTION_SIZE):
        total += s._actions[i]
    if total == 0:
        s.terminal = True
        s.terminal_value = s.player * -1
    else:
        s.terminal = False
        s.terminal_value = 0
    return s


cdef CSantoriniState _step(CSantoriniState state, int action):
    """Apply action and return new state. No np.copy or deepcopy."""
    cdef CSantoriniState ns = CSantoriniState.__new__(CSantoriniState)
    cdef int pi = _player_idx(state.player)
    cdef int w_idx, m_dir, b_dir
    cdef int wr, wc_val, mr, mc, br, bc
    cdef int old_level, new_level
    cdef bint win
    cdef int i, j, total
    cdef int sorted_wr[2]
    cdef int sorted_wc[2]

    # Decode action
    w_idx = action // 64
    m_dir = (action % 64) // 8
    b_dir = action % 8

    # Get sorted workers to find which one moves
    sorted_wr[0] = state._wr[pi][0]; sorted_wc[0] = state._wc[pi][0]
    sorted_wr[1] = state._wr[pi][1]; sorted_wc[1] = state._wc[pi][1]
    _sort_workers(sorted_wr, sorted_wc)

    wr = sorted_wr[w_idx]
    wc_val = sorted_wc[w_idx]
    mr = wr + DIR_R[m_dir]
    mc = wc_val + DIR_C[m_dir]
    br = mr + DIR_R[b_dir]
    bc = mc + DIR_C[b_dir]

    # Check win (climbed to level 3)
    old_level = state._board[wr][wc_val]
    new_level = state._board[mr][mc]
    win = (new_level == 3 and old_level < 3)

    # Copy board
    memcpy(ns._board, state._board, 25 * sizeof(int))

    # Copy workers
    for i in range(2):
        for j in range(2):
            ns._wr[i][j] = state._wr[i][j]
            ns._wc[i][j] = state._wc[i][j]

    # Move worker: find the original (unsorted) worker that matches
    for i in range(2):
        if state._wr[pi][i] == wr and state._wc[pi][i] == wc_val:
            ns._wr[pi][i] = mr
            ns._wc[pi][i] = mc
            break

    # Build
    ns._board[br][bc] += 1

    ns.player = state.player * -1
    ns.prev_state = state
    ns.last_turn_skipped = False
    ns._board_np = None
    ns._actions_np = None

    if win:
        memset(ns._actions, 0, ACTION_SIZE * sizeof(int))
        ns.terminal = True
        ns.terminal_value = state.player  # The player who moved wins
    else:
        _compute_available(ns)
        # Check if current player (ns.player) has no moves
        total = 0
        for i in range(ACTION_SIZE):
            total += ns._actions[i]
        if total == 0:
            ns.terminal = True
            ns.terminal_value = ns.player * -1  # Opponent wins
        else:
            ns.terminal = False
            ns.terminal_value = 0

    return ns


class CSantoriniGame:
    """Cython-accelerated Santorini game — same interface as SantoriniGame."""
    board_shape = (BOARD_SIZE, BOARD_SIZE)
    action_size = ACTION_SIZE
    num_history_states = 0
    input_channels = 7

    def new_game(self):
        return _new_game()

    def step(self, state, int action):
        if action < 0 or action >= ACTION_SIZE:
            raise ValueError(f"Invalid action {action}")
        if state.available_actions[action] != 1:
            raise ValueError(f"Action {action} not available")
        return _step(state, action)

    def state_to_input(self, state):
        cdef cnp.ndarray[float, ndim=3] inp = np.zeros(
            (7, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        cdef float[:, :, :] out = inp
        cdef int r, c, level, pi, oi
        cdef CSantoriniState cs

        if isinstance(state, CSantoriniState):
            cs = <CSantoriniState>state
            # Channels 0-4: one-hot building levels
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    level = cs._board[r][c]
                    if level < 5:
                        out[level, r, c] = 1.0
                    else:
                        out[4, r, c] = 1.0  # 4+ = dome

            # Channel 5: current player's workers
            pi = _player_idx(cs.player)
            out[5, cs._wr[pi][0], cs._wc[pi][0]] = 1.0
            out[5, cs._wr[pi][1], cs._wc[pi][1]] = 1.0

            # Channel 6: opponent's workers
            oi = 1 - pi
            out[6, cs._wr[oi][0], cs._wc[oi][0]] = 1.0
            out[6, cs._wr[oi][1], cs._wc[oi][1]] = 1.0
        else:
            # Fallback for Python GameState
            board = state.board
            for level in range(5):
                inp[level] = (board == level).astype(np.float32)
            for r2, c2 in state.workers[state.player]:
                inp[5][r2][c2] = 1.0
            opponent = state.player * -1
            for r2, c2 in state.workers[opponent]:
                inp[6][r2][c2] = 1.0

        return inp

    def get_symmetries(self, state_input, policy):
        """No symmetry augmentation for Santorini (action encoding is direction-based)."""
        return [(state_input, policy)]
