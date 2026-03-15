import numpy as np
from .base import GameState as BaseGameState, Game


class GameState(BaseGameState):
    def __init__(self, prev_state, board=None, player=-1):
        if board is None:
            board = np.zeros((3, 3), dtype="int")
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
        if np.count_nonzero(self.board) == 9:
            return True, 0
        return False, None

    def _available_actions(self):
        mask = np.zeros(9, dtype="int")
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 0:
                    mask[3 * row + col] = 1
        return mask

    def _check_player_won(self, player):
        b = self.board
        for r in range(3):
            if b[r][0] == player and b[r][1] == player and b[r][2] == player:
                return True
        for c in range(3):
            if b[0][c] == player and b[1][c] == player and b[2][c] == player:
                return True
        if b[0][0] == player and b[1][1] == player and b[2][2] == player:
            return True
        if b[0][2] == player and b[1][1] == player and b[2][0] == player:
            return True
        return False


class TTTGame(Game):
    board_shape = (3, 3)
    action_size = 9
    num_history_states = 0

    def new_game(self):
        return GameState(None)

    def step(self, state, action):
        if action < 0 or action > 8:
            raise ValueError(f"Invalid action {action}")
        x = action // 3
        y = action % 3
        if state.board[x][y] != 0:
            raise ValueError(f"Invalid action, ({x},{y}) is occupied")
        next_board = np.copy(state.board)
        next_board[x][y] = state.player
        return GameState(state, next_board, state.player * -1)

    input_channels = 3

    def state_to_input(self, state):
        inp = np.zeros((3, 3, 3), dtype="float32")
        me = state.player
        inp[0] = (state.board == me).astype("float32")
        inp[1] = (state.board == -me).astype("float32")
        inp[2] = me  # current player plane: +1 or -1
        return inp
