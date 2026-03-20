import numpy as np
from .base import GameState as BaseGameState, Game

ROW_COUNT = 6
COLUMN_COUNT = 7


class GameState(BaseGameState):
    def __init__(self, prev_state, board=None, player=-1):
        if board is None:
            board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
        self.board = board
        self.player = player
        self.available_actions = self._available_actions()
        self.terminal, self.terminal_value = self._over()
        self.last_turn_skipped = False
        self.prev_state = prev_state

    def _over(self):
        if self._check_player_won(-1):
            return True, -1
        if self._check_player_won(1):
            return True, 1
        if np.count_nonzero(self.board) == ROW_COUNT * COLUMN_COUNT:
            return True, 0
        return False, None

    def _available_actions(self):
        mask = np.zeros(COLUMN_COUNT, dtype="int")
        for col in range(COLUMN_COUNT):
            if self._get_first_free_row(col) is not None:
                mask[col] = 1
        return mask

    def _get_first_free_row(self, column):
        for row in range(ROW_COUNT):
            if self.board[row][column] == 0:
                return row
        return None

    def _check_player_won(self, player):
        b = self.board
        for c in range(COLUMN_COUNT - 3):  # Horizontal
            for r in range(ROW_COUNT):
                if b[r][c] == player and b[r][c+1] == player and b[r][c+2] == player and b[r][c+3] == player:
                    return True
        for c in range(COLUMN_COUNT):  # Vertical
            for r in range(ROW_COUNT - 3):
                if b[r][c] == player and b[r+1][c] == player and b[r+2][c] == player and b[r+3][c] == player:
                    return True
        for c in range(COLUMN_COUNT - 3):  # Diagonal /
            for r in range(ROW_COUNT - 3):
                if b[r][c] == player and b[r+1][c+1] == player and b[r+2][c+2] == player and b[r+3][c+3] == player:
                    return True
        for c in range(COLUMN_COUNT - 3):  # Diagonal \
            for r in range(3, ROW_COUNT):
                if b[r][c] == player and b[r-1][c+1] == player and b[r-2][c+2] == player and b[r-3][c+3] == player:
                    return True
        return False


def compute_threat_map(board, player):
    """Compute per-cell threat map from current player's perspective.

    Returns a (6,7) float32 array where:
      +1 = placing current player's piece here completes 4-in-a-row (my threat)
      -1 = placing opponent's piece here completes 4-in-a-row (opp threat)
       0 = no threat (or cell occupied)

    If a cell is a threat for both players, +1 takes priority (canonical).
    Computable from board state alone — works with random data and self-play.
    """
    threat = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.float32)
    opp = -player

    # Check all 4-in-a-row windows
    # Directions: horizontal (0,1), vertical (1,0), diag-up (1,1), diag-down (-1,1)
    directions = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for dr, dc in directions:
        # Compute valid starting positions for this direction
        r_start = max(0, 0 - 3 * dr) if dr >= 0 else max(0, -3 * dr)
        r_end = ROW_COUNT if dr <= 0 else ROW_COUNT - 3 * dr
        c_start = 0
        c_end = COLUMN_COUNT - 3 * dc  # dc is always positive

        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                # Collect the 4 cells in this window
                cells = []
                for i in range(4):
                    cells.append((r + i * dr, c + i * dc))

                values = [board[cr][cc] for cr, cc in cells]

                # Check if exactly 3 of one color + 1 empty
                for p, sign in [(player, 1.0), (opp, -1.0)]:
                    count_p = sum(1 for v in values if v == p)
                    empty_cells = [(cr, cc) for (cr, cc), v in zip(cells, values) if v == 0]
                    if count_p == 3 and len(empty_cells) == 1:
                        er, ec = empty_cells[0]
                        # +1 takes priority over -1
                        if sign > 0 or threat[er][ec] == 0:
                            threat[er][ec] = sign

    return threat


class Connect4Game(Game):
    board_shape = (ROW_COUNT, COLUMN_COUNT)
    action_size = COLUMN_COUNT
    num_history_states = 0

    def new_game(self):
        return GameState(None)

    def step(self, state, action):
        if action < 0 or action >= COLUMN_COUNT:
            raise ValueError(f"Invalid column {action}")
        row = state._get_first_free_row(action)
        if row is None:
            raise ValueError(f"Column {action} is full")
        next_board = np.copy(state.board)
        next_board[row][action] = state.player
        return GameState(state, next_board, state.player * -1)

    def get_symmetries(self, state_input, policy, aux_maps=None):
        """Connect4 has left-right mirror symmetry."""
        syms = [(state_input, policy, aux_maps)]
        # Flip columns: board channels flip along axis 2, policy reverses
        flipped_input = state_input[:, :, ::-1].copy()
        flipped_policy = policy[::-1].copy()
        flipped_aux = None
        if aux_maps is not None:
            flipped_aux = {}
            for k, v in aux_maps.items():
                flipped_aux[k] = v[:, ::-1].copy() if v is not None else None
        syms.append((flipped_input, flipped_policy, flipped_aux))
        return syms

    def compute_threat_map(self, state):
        """Compute per-cell threat map from current player's perspective."""
        return compute_threat_map(state.board, state.player)

    input_channels = 2

    def state_to_input(self, state):
        rows, cols = self.board_shape
        inp = np.zeros((2, rows, cols), dtype="float32")
        me = state.player
        inp[0] = (state.board == me).astype("float32")   # my pieces
        inp[1] = (state.board == -me).astype("float32")   # opponent pieces
        return inp
