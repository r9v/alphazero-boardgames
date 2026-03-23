# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated Connect4 with bitboard representation.

Replaces Python GameState/Connect4Game with C-typed operations:
- Board stored as 2 uint64 bitmasks (one per player, 42 bits each)
- Win check: 4 shift+AND ops instead of 4 nested Python loops (~100x faster)
- Step: single bit-set instead of np.copy (~50x faster)
- Available actions: top-row bitmask check (~20x faster)
- state_to_input cached on the state object (avoids recomputation in MCTS)

Bitboard layout (column-major, bottom to top, with guard bit per column):
  Col 0: bits 0-5   (row 0 at bit 0, row 5 at bit 5, guard at bit 6)
  Col 1: bits 7-12  (row 0 at bit 7, ...)
  Col 2: bits 14-19
  Col 3: bits 21-26
  Col 4: bits 28-33
  Col 5: bits 35-40
  Col 6: bits 42-47

Each column uses 7 bits (6 rows + 1 guard bit for overflow detection).
Total: 49 bits, fits in uint64.
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport uint64_t

cnp.import_array()

DEF ROW_COUNT = 6
DEF COLUMN_COUNT = 7
DEF BITS_PER_COL = 7  # 6 rows + 1 guard bit

# Column base bit positions
cdef uint64_t COL_BASE[7]
COL_BASE[0] = 0
COL_BASE[1] = 7
COL_BASE[2] = 14
COL_BASE[3] = 21
COL_BASE[4] = 28
COL_BASE[5] = 35
COL_BASE[6] = 42

# Top row mask: bit for row 5 (topmost playable) in each column
cdef uint64_t TOP_ROW_MASK = 0
cdef int _i
for _i in range(COLUMN_COUNT):
    TOP_ROW_MASK |= (<uint64_t>1 << (COL_BASE[_i] + ROW_COUNT - 1))

# Full board mask: all 42 playable bits
cdef uint64_t FULL_BOARD_MASK = 0
cdef int _r, _c
for _c in range(COLUMN_COUNT):
    for _r in range(ROW_COUNT):
        FULL_BOARD_MASK |= (<uint64_t>1 << (COL_BASE[_c] + _r))


cdef inline bint _check_win(uint64_t bb) noexcept nogil:
    """Check if a single-player bitboard contains 4-in-a-row.

    Uses the standard bitboard trick: shift and AND. If bb & (bb>>d) & (bb>>2d) & (bb>>3d)
    is nonzero, there's a 4-in-a-row in direction d.

    Directions (shift amounts):
      Vertical: 1 (adjacent rows in same column)
      Horizontal: 7 (adjacent columns in same row)
      Diagonal /: 8 (row+1, col+1)
      Diagonal \\: 6 (row-1, col+1)
    """
    cdef uint64_t m
    # Vertical (shift by 1)
    m = bb & (bb >> 1)
    if m & (m >> 2):
        return True
    # Horizontal (shift by BITS_PER_COL=7)
    m = bb & (bb >> 7)
    if m & (m >> 14):
        return True
    # Diagonal / (shift by BITS_PER_COL+1=8)
    m = bb & (bb >> 8)
    if m & (m >> 16):
        return True
    # Diagonal \ (shift by BITS_PER_COL-1=6)
    m = bb & (bb >> 6)
    if m & (m >> 12):
        return True
    return False


cdef inline int _col_height(uint64_t both, int col) noexcept nogil:
    """Return number of pieces in column (0-6)."""
    cdef uint64_t col_bits = both >> COL_BASE[col]
    cdef int h = 0
    while h < ROW_COUNT and (col_bits & 1):
        h += 1
        col_bits >>= 1
    return h


cdef class CConnect4State:
    """Connect4 game state with bitboard internals."""
    cdef public uint64_t bb_me     # bitboard for current player's pieces
    cdef public uint64_t bb_opp    # bitboard for opponent's pieces
    cdef public int player         # -1 or 1
    cdef public bint terminal
    cdef public int terminal_value
    cdef public bint last_turn_skipped
    cdef public object prev_state

    # Column heights (0-6 per column), avoids recomputation
    cdef int _heights[7]

    # Cached numpy objects (created on demand)
    cdef object _board_np
    cdef object _avail_np
    cdef object _input_np

    def __cinit__(self):
        self._board_np = None
        self._avail_np = None
        self._input_np = None

    @staticmethod
    cdef CConnect4State create(uint64_t bb_me, uint64_t bb_opp,
                                int player, object prev_state):
        """Fast C-level constructor."""
        cdef CConnect4State s = CConnect4State.__new__(CConnect4State)
        s.bb_me = bb_me
        s.bb_opp = bb_opp
        s.player = player
        s.prev_state = prev_state
        s.last_turn_skipped = False
        s._board_np = None
        s._avail_np = None
        s._input_np = None

        # Compute column heights from combined bitboard
        cdef uint64_t both = bb_me | bb_opp
        cdef int c
        for c in range(COLUMN_COUNT):
            s._heights[c] = _col_height(both, c)

        # Check terminal conditions
        # The LAST player to move is the opponent (we alternate after step)
        if _check_win(bb_opp):
            # Opponent just moved and won
            s.terminal = True
            s.terminal_value = -player  # opponent is -player in absolute terms...
            # Actually: player field is who moves NEXT. The one who just moved is -player.
            # terminal_value convention: -1 = X won, +1 = O won
            # The piece just placed belongs to -player (the one who just moved).
            # If -player's bitboard has a win, then -player won.
            # But wait: bb_opp is from current player's perspective.
            # After step(), we swap: new bb_me = old bb_opp, new bb_opp = old bb_me + new piece.
            # So bb_opp is the pieces of the player who just moved.
            s.terminal_value = -player  # -player just won in absolute coords
        elif (bb_me | bb_opp) == FULL_BOARD_MASK:
            s.terminal = True
            s.terminal_value = 0  # draw
        else:
            s.terminal = False
            s.terminal_value = 0

        return s

    def __init__(self):
        """Python-level init for empty board."""
        self.bb_me = 0
        self.bb_opp = 0
        self.player = -1  # X goes first
        self.prev_state = None
        self.last_turn_skipped = False
        self.terminal = False
        self.terminal_value = 0
        cdef int c
        for c in range(COLUMN_COUNT):
            self._heights[c] = 0

    @property
    def board(self):
        """Numpy board (6,7) for compatibility. Cached after first access."""
        if self._board_np is not None:
            return self._board_np
        cdef cnp.ndarray[int, ndim=2] b = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.intc)
        cdef int r, c
        cdef uint64_t bit
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                bit = <uint64_t>1 << (COL_BASE[c] + r)
                if self.bb_me & bit:
                    b[r, c] = self.player
                elif self.bb_opp & bit:
                    b[r, c] = -self.player
        self._board_np = b
        return b

    @property
    def available_actions(self):
        """Binary mask array (7,) of legal columns. Cached."""
        if self._avail_np is not None:
            return self._avail_np
        cdef cnp.ndarray[int, ndim=1] mask = np.zeros(COLUMN_COUNT, dtype=np.intc)
        cdef int c
        for c in range(COLUMN_COUNT):
            if self._heights[c] < ROW_COUNT:
                mask[c] = 1
        self._avail_np = mask
        return mask

    cdef inline bint _col_available(self, int col) noexcept:
        return self._heights[col] < ROW_COUNT

    def _get_first_free_row(self, int column):
        """Compatibility: return first free row or None."""
        if self._heights[column] < ROW_COUNT:
            return self._heights[column]
        return None

    def get_cached_input(self):
        """Return cached state_to_input result, or None if not cached yet."""
        return self._input_np

    def set_cached_input(self, inp):
        """Cache state_to_input result on this state."""
        self._input_np = inp


def _from_numpy_board(board, int player, prev_state=None):
    """Create a CConnect4State from a numpy (6,7) board and player."""
    cdef uint64_t bb_me = 0, bb_opp = 0
    cdef int r, c
    cdef uint64_t bit
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            bit = <uint64_t>1 << (COL_BASE[c] + r)
            if board[r][c] == player:
                bb_me |= bit
            elif board[r][c] == -player:
                bb_opp |= bit
    return CConnect4State.create(bb_me, bb_opp, player, prev_state)


cdef class CConnect4Game:
    """Cython Connect4 game — drop-in replacement for Python Connect4Game."""

    cdef public tuple board_shape
    cdef public int action_size
    cdef public int num_history_states
    cdef public int input_channels

    def __init__(self):
        self.board_shape = (ROW_COUNT, COLUMN_COUNT)
        self.action_size = COLUMN_COUNT
        self.num_history_states = 0
        self.input_channels = 2

    def new_game(self):
        return CConnect4State()

    def step(self, state, int action):
        if not isinstance(state, CConnect4State):
            # Convert Python GameState to CConnect4State
            state = _from_numpy_board(state.board, state.player, state)

        cdef CConnect4State cs = <CConnect4State>state
        if action < 0 or action >= COLUMN_COUNT:
            raise ValueError(f"Invalid column {action}")
        if cs._heights[action] >= ROW_COUNT:
            raise ValueError(f"Column {action} is full")

        cdef int row = cs._heights[action]
        cdef uint64_t bit = <uint64_t>1 << (COL_BASE[action] + row)

        cdef uint64_t new_bb_me = cs.bb_opp
        cdef uint64_t new_bb_opp = cs.bb_me | bit
        cdef int new_player = -cs.player

        return CConnect4State.create(new_bb_me, new_bb_opp, new_player, cs)

    def state_to_input(self, state):
        """Encode state as (2, 6, 7) float32 tensor. Caches on CConnect4State."""
        if isinstance(state, CConnect4State):
            return self._state_to_input_fast(<CConnect4State>state)
        # Fallback for Python GameState (used by training logger for fixed positions)
        cdef cnp.ndarray[float, ndim=3] inp = np.zeros((2, ROW_COUNT, COLUMN_COUNT), dtype=np.float32)
        me = state.player
        inp[0] = (state.board == me).astype(np.float32)
        inp[1] = (state.board == -me).astype(np.float32)
        return inp

    cdef object _state_to_input_fast(self, CConnect4State state):
        """Fast path for CConnect4State with bitboard encoding + caching."""
        cdef object cached = state._input_np
        if cached is not None:
            return cached

        cdef cnp.ndarray[float, ndim=3] inp = np.zeros((2, ROW_COUNT, COLUMN_COUNT), dtype=np.float32)
        cdef int r, c
        cdef uint64_t bit
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                bit = <uint64_t>1 << (COL_BASE[c] + r)
                if state.bb_me & bit:
                    inp[0, r, c] = 1.0
                elif state.bb_opp & bit:
                    inp[1, r, c] = 1.0

        state._input_np = inp
        return inp

    def compute_threat_map(self, state):
        """Compute per-cell threat map from current player's perspective.

        Returns (6,7) float32: +1=my threat, -1=opp threat, 0=none.
        Uses bitboard operations for speed.
        """
        # Fallback for Python GameState
        if not isinstance(state, CConnect4State):
            return compute_threat_map(state.board, state.player)

        cdef cnp.ndarray[float, ndim=2] threat = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.float32)
        cdef uint64_t my_bb = (<CConnect4State>state).bb_me
        cdef uint64_t opp_bb = (<CConnect4State>state).bb_opp
        cdef uint64_t both = my_bb | opp_bb
        cdef uint64_t empty
        cdef int r, c, dr, dc, i
        cdef int cr, cc
        cdef int count_me, count_opp, count_empty
        cdef int er, ec

        # Directions: horizontal(0,1), vertical(1,0), diag-up(1,1), diag-down(-1,1)
        cdef int dirs_r[4]
        cdef int dirs_c[4]
        dirs_r[0] = 0; dirs_c[0] = 1
        dirs_r[1] = 1; dirs_c[1] = 0
        dirs_r[2] = 1; dirs_c[2] = 1
        dirs_r[3] = -1; dirs_c[3] = 1

        cdef int cells_r[4]
        cdef int cells_c[4]
        cdef int vals[4]  # 1=me, -1=opp, 0=empty
        cdef uint64_t bit

        cdef int d, r_start, r_end, c_end
        for d in range(4):
            dr = dirs_r[d]
            dc = dirs_c[d]
            # Valid starting positions
            if dr > 0:
                r_start = 0
                r_end = ROW_COUNT - 3 * dr
            elif dr < 0:
                r_start = -3 * dr
                r_end = ROW_COUNT
            else:
                r_start = 0
                r_end = ROW_COUNT
            c_end = COLUMN_COUNT - 3 * dc

            for r in range(r_start, r_end):
                for c in range(0, c_end):
                    # Collect 4 cells
                    count_me = 0
                    count_opp = 0
                    count_empty = 0
                    er = -1
                    ec = -1
                    for i in range(4):
                        cr = r + i * dr
                        cc = c + i * dc
                        bit = <uint64_t>1 << (COL_BASE[cc] + cr)
                        if my_bb & bit:
                            count_me += 1
                        elif opp_bb & bit:
                            count_opp += 1
                        else:
                            count_empty += 1
                            er = cr
                            ec = cc

                    if count_empty == 1:
                        if count_me == 3:
                            threat[er, ec] = 1.0  # my threat (priority)
                        elif count_opp == 3 and threat[er, ec] == 0.0:
                            threat[er, ec] = -1.0  # opp threat

        return threat

    def get_symmetries(self, state_input, policy, aux_maps=None):
        """Connect4 left-right mirror symmetry."""
        syms = [(state_input, policy, aux_maps)]
        flipped_input = state_input[:, :, ::-1].copy()
        flipped_policy = policy[::-1].copy()
        flipped_aux = None
        if aux_maps is not None:
            flipped_aux = {}
            for k, v in aux_maps.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    flipped_aux[k] = v[:, ::-1].copy() if v is not None else None
                else:
                    flipped_aux[k] = v
        syms.append((flipped_input, flipped_policy, flipped_aux))
        return syms


def classify_win(board):
    """Classify how the game was won. Returns list of (type, col, player) tuples.

    Compatible with the Python version — accepts numpy board (6,7).
    """
    wins = []
    cdef int r, c
    for player_int in [-1, 1]:
        # Horizontal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if (board[r][c] == player_int and board[r][c+1] == player_int and
                    board[r][c+2] == player_int and board[r][c+3] == player_int):
                    wins.append(('horiz', c, player_int))
        # Vertical
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == player_int and board[r+1][c] == player_int and
                    board[r+2][c] == player_int and board[r+3][c] == player_int):
                    wins.append(('vert', c, player_int))
        # Diagonal /
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if (board[r][c] == player_int and board[r+1][c+1] == player_int and
                    board[r+2][c+2] == player_int and board[r+3][c+3] == player_int):
                    wins.append(('diag_up', c, player_int))
        # Diagonal \
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if (board[r][c] == player_int and board[r-1][c+1] == player_int and
                    board[r-2][c+2] == player_int and board[r-3][c+3] == player_int):
                    wins.append(('diag_down', c, player_int))
    return wins


def compute_threat_map(board, int player):
    """Standalone threat map function compatible with Python version.

    Accepts numpy board (6,7) and player (-1 or 1).
    """
    # Convert numpy board to bitboards, then use Cython
    cdef uint64_t bb_me = 0, bb_opp = 0
    cdef int r, c
    cdef uint64_t bit
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            bit = <uint64_t>1 << (COL_BASE[c] + r)
            if board[r][c] == player:
                bb_me |= bit
            elif board[r][c] == -player:
                bb_opp |= bit

    # Create a temporary state for the method
    cdef CConnect4State s = CConnect4State.__new__(CConnect4State)
    s.bb_me = bb_me
    s.bb_opp = bb_opp
    s.player = player

    cdef CConnect4Game game = CConnect4Game()
    return game.compute_threat_map(s)
