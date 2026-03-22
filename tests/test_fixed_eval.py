"""Tests for FixedEval diagnostic positions.

Validates that all hand-crafted positions are:
1. Gravity-valid (no floating pieces)
2. Have correct piece counts (equal for X-to-move, X+1 for O-to-move)
3. The expected winning move actually wins the game
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from games.connect4 import Connect4Game, GameState as C4State

game = Connect4Game()

ROW_COUNT = 6
COLUMN_COUNT = 7


def check_gravity(board, name):
    """Verify no floating pieces — every piece must have support below it."""
    for col in range(COLUMN_COUNT):
        found_empty = False
        for row in range(ROW_COUNT):
            if board[row][col] == 0:
                found_empty = True
            elif found_empty:
                # Piece above an empty cell = floating
                assert False, (
                    f"{name}: floating piece at row={row}, col={col}. "
                    f"Column {col} contents (bottom to top): "
                    f"{[board[r][col] for r in range(ROW_COUNT)]}"
                )


def check_piece_counts(board, player, name):
    """Verify piece counts are consistent with whose turn it is.

    X=-1 moves first. If X to move: count(X) == count(O).
    If O to move: count(X) == count(O) + 1.
    """
    x_count = np.count_nonzero(board == -1)
    o_count = np.count_nonzero(board == 1)
    if player == -1:  # X to move
        assert x_count == o_count, (
            f"{name}: X to move but X={x_count}, O={o_count} (should be equal)"
        )
    else:  # O to move
        assert x_count == o_count + 1, (
            f"{name}: O to move but X={x_count}, O={o_count} (should be X=O+1)"
        )


def check_not_already_terminal(board, player, name):
    """Verify the position isn't already won/drawn before the winning move."""
    s = C4State(None, board.copy(), player=player)
    assert not s.terminal, (
        f"{name}: position is already terminal before winning move. "
        f"terminal_value={s.terminal_value}"
    )


def check_winning_move(board, player, winning_col, name):
    """Verify playing the winning column actually wins."""
    s = C4State(None, board.copy(), player=player)
    assert s.available_actions[winning_col], (
        f"{name}: winning col {winning_col} is not available"
    )
    ns = game.step(s, winning_col)
    assert ns.terminal, (
        f"{name}: playing col {winning_col} does not end the game"
    )
    assert ns.terminal_value == player, (
        f"{name}: playing col {winning_col} ends game but terminal_value="
        f"{ns.terminal_value}, expected {player} (current player wins)"
    )


def build_positions():
    """Build all FixedEval positions (mirrors training_logger._get_diagnostic_positions)."""
    positions = []

    # empty_board
    board = np.zeros((6, 7), dtype="int")
    positions.append(("empty_board", board, -1, None, "~0"))

    # x_wins_next: X has 3 horizontal at cols 0-2, plays col 3
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = -1
    board[0][4] = 1
    board[0][5] = 1
    board[1][4] = 1
    positions.append(("x_wins_next", board, -1, 3, "> +0.5"))

    # o_wins_next: O has 3 horizontal at cols 4-6, plays col 3
    board = np.zeros((6, 7), dtype="int")
    board[0][0:3] = 1
    board[0][4:7] = -1
    board[1][4] = -1
    positions.append(("o_wins_next", board, 1, 3, "> +0.5"))

    # diag_threat: complex mid-game, O to move
    board = np.zeros((6, 7), dtype="int")
    board[0] = [0, 1, 1, -1, 0, 1, -1]
    board[1] = [0, -1, 1, -1, 0, 0, 0]
    board[2] = [0, 1, -1, 1, 0, 0, 0]
    board[3] = [0, -1, 0, -1, 0, 0, 0]
    positions.append(("diag_threat", board, 1, None, "< 0"))

    # x_center: X controls center
    board = np.zeros((6, 7), dtype="int")
    board[0][3] = -1
    board[1][3] = -1
    board[0][2] = 1
    board[0][4] = 1
    positions.append(("x_center", board, -1, None, "> 0"))

    # vert_wins: X has 3 stacked in col 3, plays col 3
    board = np.zeros((6, 7), dtype="int")
    board[0][3] = -1
    board[1][3] = -1
    board[2][3] = -1
    board[0][1] = 1
    board[0][5] = 1
    board[0][6] = 1
    positions.append(("vert_wins", board, -1, 3, "> +0.5"))

    # horiz_right: X has 3 at cols 4-6, plays col 3
    board = np.zeros((6, 7), dtype="int")
    board[0][4] = -1
    board[0][5] = -1
    board[0][6] = -1
    board[0][0] = 1
    board[0][1] = 1
    board[1][0] = 1
    positions.append(("horiz_right", board, -1, 3, "> +0.5"))

    # vert_edge: X has 3 stacked in col 0, plays col 0 (O has 3-in-a-row threat)
    board = np.zeros((6, 7), dtype="int")
    board[0][0] = -1
    board[1][0] = -1
    board[2][0] = -1
    board[0][3] = 1
    board[0][4] = 1
    board[0][5] = 1
    positions.append(("vert_edge", board, -1, 0, "> +0.5"))

    # vert_edge_clean: same but O scattered (no O threat)
    board = np.zeros((6, 7), dtype="int")
    board[0][0] = -1
    board[1][0] = -1
    board[2][0] = -1
    board[0][2] = 1
    board[0][4] = 1
    board[0][6] = 1
    positions.append(("vert_edge_clean", board, -1, 0, "> +0.5"))

    # diag_wins: X has diagonal (0,0)(1,1)(2,2), plays col 3 for (3,3)
    board = np.zeros((6, 7), dtype="int")
    board[0][0] = -1
    board[0][1] = 1
    board[1][1] = -1
    board[0][2] = 1
    board[1][2] = -1
    board[2][2] = -1
    board[0][3] = -1
    board[1][3] = 1
    board[2][3] = 1
    board[0][4] = 1
    positions.append(("diag_wins", board, -1, 3, "> +0.5"))

    # Race: X vert col 0 vs O vert col 3, X wins by tempo
    board = np.zeros((6, 7), dtype="int")
    board[0][0] = board[1][0] = board[2][0] = -1
    board[0][3] = board[1][3] = board[2][3] = 1
    positions.append(("race_edge_ctr", board, -1, 0, "> +0.5"))

    # Horizontal win requiring edge col 0: X at cols 1,2,3 plays col 0
    board = np.zeros((6, 7), dtype="int")
    board[0][1] = board[0][2] = board[0][3] = -1
    board[0][5] = board[1][5] = board[0][6] = 1
    positions.append(("horiz_edge", board, -1, 0, "> +0.5"))

    # Race horizontal: X cols 0-2 vs O cols 4-6, X plays col 3
    board = np.zeros((6, 7), dtype="int")
    board[0][0] = board[0][1] = board[0][2] = -1
    board[0][4] = board[0][5] = board[0][6] = 1
    positions.append(("race_horiz", board, -1, 3, "> +0.5"))

    # Vertical win at right edge (col 6), O scattered
    board = np.zeros((6, 7), dtype="int")
    board[0][6] = board[1][6] = board[2][6] = -1
    board[0][0] = board[0][2] = board[0][4] = 1
    positions.append(("vert_edge_right", board, -1, 6, "> +0.5"))

    # Late game: 9X 9O, X has vert 3 at col 1, plays col 1
    board = np.zeros((6, 7), dtype="int")
    board[0] = [-1, -1,  1,  1,  1, -1,  1]
    board[1] = [ 1, -1, -1, -1,  1,  1, -1]
    board[2] = [-1, -1,  1,  1,  0,  0,  0]
    positions.append(("late_vert", board, -1, 1, "> +0.5"))

    return positions


def test_all_positions():
    positions = build_positions()
    passed = 0
    failed = 0

    for name, board, player, winning_col, expect in positions:
        try:
            check_gravity(board, name)
            check_piece_counts(board, player, name)

            if winning_col is not None:
                check_not_already_terminal(board, player, name)
                check_winning_move(board, player, winning_col, name)

            print(f"  PASS {name}: valid (player={'X' if player == -1 else 'O'}, "
                  f"win_col={winning_col}, expect {expect})")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(positions)}")
    return failed == 0


if __name__ == "__main__":
    print("Testing FixedEval diagnostic positions...\n")
    success = test_all_positions()
    sys.exit(0 if success else 1)
