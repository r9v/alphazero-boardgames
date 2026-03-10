import numpy as np
import copy
from ..base import GameState as BaseGameState, Game

BOARD_SIZE = 5

# 8 compass directions: N, NE, E, SE, S, SW, W, NW
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

INITIAL_WORKERS = {
    -1: [(0, 1), (0, 3)],
    1: [(4, 1), (4, 3)],
}


def _in_bounds(r, c):
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


class GameState(BaseGameState):
    def __init__(self, prev_state, board=None, workers=None, player=-1,
                 win_by_climb=False):
        if board is None:
            board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype="int")
        if workers is None:
            workers = copy.deepcopy(INITIAL_WORKERS)

        self.board = board
        self.workers = workers
        self.player = player
        self.prev_state = prev_state
        self.last_turn_skipped = False

        if win_by_climb:
            # Previous player won by climbing to level 3
            self.available_actions = np.zeros(128, dtype="int")
            self.terminal = True
            self.terminal_value = player * -1
        else:
            self.available_actions = self._available_actions()
            self.terminal, self.terminal_value = self._over()

    def _sorted_workers(self, player):
        """Return workers sorted by (row, col) for consistent indexing."""
        return sorted(self.workers[player])

    def _all_worker_positions(self):
        """Return set of all worker positions."""
        positions = set()
        for p in (-1, 1):
            for pos in self.workers[p]:
                positions.add(pos)
        return positions

    def _available_actions(self):
        mask = np.zeros(128, dtype="int")
        occupied = self._all_worker_positions()
        my_workers = self._sorted_workers(self.player)

        for w_idx, (wr, wc) in enumerate(my_workers):
            current_level = self.board[wr][wc]

            for m_dir, (dr, dc) in enumerate(DIRECTIONS):
                mr, mc = wr + dr, wc + dc

                # Validate move target
                if not _in_bounds(mr, mc):
                    continue
                if (mr, mc) in occupied:
                    continue
                if self.board[mr][mc] >= 4:  # dome
                    continue
                if self.board[mr][mc] - current_level > 1:  # climb too high
                    continue

                # Worker moved: compute occupied positions after move
                occupied_after = (occupied - {(wr, wc)}) | {(mr, mc)}

                for b_dir, (bdr, bdc) in enumerate(DIRECTIONS):
                    br, bc = mr + bdr, mc + bdc

                    # Validate build target
                    if not _in_bounds(br, bc):
                        continue
                    if (br, bc) in occupied_after:
                        continue
                    if self.board[br][bc] >= 4:  # dome
                        continue

                    mask[w_idx * 64 + m_dir * 8 + b_dir] = 1

        return mask

    def _over(self):
        if np.sum(self.available_actions) == 0:
            # Current player has no moves — opponent wins
            return True, self.player * -1
        return False, None


class SantoriniGame(Game):
    board_shape = (BOARD_SIZE, BOARD_SIZE)
    action_size = 128
    num_history_states = 0
    input_channels = 7

    def new_game(self):
        return GameState(None)

    def step(self, state, action):
        if action < 0 or action >= 128:
            raise ValueError(f"Invalid action {action}")
        if state.available_actions[action] != 1:
            raise ValueError(f"Action {action} not available")

        # Decode action
        w_idx = action // 64
        m_dir = (action % 64) // 8
        b_dir = action % 8

        my_workers = state._sorted_workers(state.player)
        wr, wc = my_workers[w_idx]

        dr, dc = DIRECTIONS[m_dir]
        mr, mc = wr + dr, wc + dc

        bdr, bdc = DIRECTIONS[b_dir]
        br, bc = mr + bdr, mc + bdc

        # Detect win: moved UP to level 3
        old_level = state.board[wr][wc]
        new_level = state.board[mr][mc]
        win = new_level == 3 and old_level < 3

        # Create new board and workers
        new_board = np.copy(state.board)
        new_workers = copy.deepcopy(state.workers)

        # Move worker
        w_list = new_workers[state.player]
        for i, pos in enumerate(w_list):
            if pos == (wr, wc):
                w_list[i] = (mr, mc)
                break

        # Build
        new_board[br][bc] += 1

        return GameState(state, new_board, new_workers, state.player * -1,
                         win_by_climb=win)

    def state_to_input(self, state):
        inp = np.zeros((self.input_channels, BOARD_SIZE, BOARD_SIZE),
                        dtype="float32")

        # Channels 0-4: one-hot building levels
        for level in range(5):
            inp[level] = (state.board == level).astype("float32")

        # Channel 5: current player's workers
        for r, c in state.workers[state.player]:
            inp[5][r][c] = 1.0

        # Channel 6: opponent's workers
        opponent = state.player * -1
        for r, c in state.workers[opponent]:
            inp[6][r][c] = 1.0

        return inp
