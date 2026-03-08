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


class Connect4Game(Game):
    board_shape = (ROW_COUNT, COLUMN_COUNT)
    action_size = COLUMN_COUNT
    num_history_states = 2

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

    def state_to_input(self, state):
        rows, cols = self.board_shape
        channels = 2 * (self.num_history_states + 1) + 2  # 8
        inp = np.zeros((channels, rows, cols), dtype="float32")

        s = state
        history = []
        for _ in range(self.num_history_states):
            if s.prev_state is not None:
                s = s.prev_state
                history.append(s.board)
            else:
                history.append(np.zeros((rows, cols), dtype="int"))
        history.reverse()

        for i, board in enumerate(history):
            inp[2 * i] = (board == -1).astype("float32")
            inp[2 * i + 1] = (board == 1).astype("float32")

        c = 2 * self.num_history_states
        inp[c] = (state.board == -1).astype("float32")
        inp[c + 1] = (state.board == 1).astype("float32")

        if state.player == -1:
            inp[c + 2] = 1.0
        else:
            inp[c + 3] = 1.0

        return inp
