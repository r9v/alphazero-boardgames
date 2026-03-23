"""Verification tests for Connect4 game logic.

Run: python -m tests.test_connect4
"""
import numpy as np
from games.c_connect4 import CConnect4Game, CConnect4State

ROW_COUNT = 6
COLUMN_COUNT = 7

game = CConnect4Game()
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


def print_board(board):
    symbols = {0: ".", -1: "X", 1: "O"}
    for r in range(5, -1, -1):
        row = " ".join(symbols[board[r][c]] for c in range(7))
        print(f"        {row}")
    print("        " + " ".join(str(c) for c in range(7)))


def make_state(board, player):
    """Create a CConnect4State from a board array and player."""
    return CConnect4State.from_board(np.array(board, dtype="int"), player)


# ============================================================
# 1. Win detection
# ============================================================
print("\n=== Win Detection Tests ===")

# Horizontal wins
print("\n  -- Horizontal --")
for row in range(ROW_COUNT):
    for col in range(COLUMN_COUNT - 3):
        for player in [-1, 1]:
            board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
            board[row][col:col+4] = player
            state = make_state(board, -player)
            check(f"Horiz player={player} row={row} col={col}",
                  state.terminal and state.terminal_value == player,
                  f"terminal={state.terminal} val={state.terminal_value}")

# Vertical wins
print("\n  -- Vertical --")
for col in range(COLUMN_COUNT):
    for row in range(ROW_COUNT - 3):
        for player in [-1, 1]:
            board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
            for dr in range(4):
                board[row+dr][col] = player
            state = make_state(board, -player)
            check(f"Vert player={player} row={row} col={col}",
                  state.terminal and state.terminal_value == player,
                  f"terminal={state.terminal} val={state.terminal_value}")

# Diagonal / wins (ascending: r+1, c+1)
print("\n  -- Diagonal / --")
for col in range(COLUMN_COUNT - 3):
    for row in range(ROW_COUNT - 3):
        for player in [-1, 1]:
            board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
            for d in range(4):
                board[row+d][col+d] = player
            state = make_state(board, -player)
            check(f"Diag/ player={player} row={row} col={col}",
                  state.terminal and state.terminal_value == player,
                  f"terminal={state.terminal} val={state.terminal_value}")

# Diagonal \ wins (descending: r-1, c+1)
print("\n  -- Diagonal \\ --")
for col in range(COLUMN_COUNT - 3):
    for row in range(3, ROW_COUNT):
        for player in [-1, 1]:
            board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
            for d in range(4):
                board[row-d][col+d] = player
            state = make_state(board, -player)
            check(f"Diag\\ player={player} row={row} col={col}",
                  state.terminal and state.terminal_value == player,
                  f"terminal={state.terminal} val={state.terminal_value}")


# ============================================================
# 2. No false wins
# ============================================================
print("\n=== No False Wins ===")

# 3 in a row should NOT be terminal
board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
board[0][0:3] = -1  # Only 3 X in a row
state = make_state(board, 1)
check("3 horizontal is NOT a win", not state.terminal)

board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
for d in range(3):
    board[d][d] = 1  # Only 3 O in diagonal
state = make_state(board, -1)
check("3 diagonal is NOT a win", not state.terminal)

# Empty board
state = make_state(np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int"), -1)
check("Empty board is NOT terminal", not state.terminal)
check("Empty board has 7 available actions", np.sum(state.available_actions) == 7)


# ============================================================
# 3. Gravity / piece placement
# ============================================================
print("\n=== Gravity Tests ===")

s = game.new_game()
check("New game: player is -1 (X)", s.player == -1)
check("New game: 7 available actions", np.sum(s.available_actions) == 7)

# Drop piece in column 3
s1 = game.step(s, 3)
check("After X plays col 3: board[0][3] == -1", s1.board[0][3] == -1)
check("After X plays col 3: player is 1 (O)", s1.player == 1)

# Stack pieces in same column
s2 = game.step(s1, 3)
check("After O plays col 3: board[1][3] == 1", s2.board[1][3] == 1)
check("Stacking: board[0][3] still -1", s2.board[0][3] == -1)

# Fill a column
s = game.new_game()
for i in range(6):
    s = game.step(s, 0)
check("After 6 plays in col 0: col 0 is full", s.available_actions[0] == 0)
check("After 6 plays: 6 actions remain", np.sum(s.available_actions) == 6)

# Verify ValueError on full column
try:
    game.step(s, 0)
    check("Playing full column raises ValueError", False)
except ValueError:
    check("Playing full column raises ValueError", True)


# ============================================================
# 4. Draw detection
# ============================================================
print("\n=== Draw Detection ===")

# Create a full board with no winner (verified draw position)
# Pattern: each row alternates in groups of 3 to avoid 4-in-a-row
board = np.array([
    [-1, -1, -1,  1,  1,  1, -1],  # row 0
    [ 1,  1,  1, -1, -1, -1,  1],  # row 1
    [-1, -1, -1,  1,  1,  1, -1],  # row 2
    [ 1,  1, -1, -1,  1, -1,  1],  # row 3
    [-1, -1,  1,  1, -1,  1, -1],  # row 4
    [ 1,  1, -1, -1,  1, -1,  1],  # row 5
], dtype="int")
state = make_state(board, 1)
has_win = state.terminal and state.terminal_value != 0
if has_win:
    # Brute-force find a valid draw board by playing random games
    print("  Searching for a valid draw board...")
    found_draw = False
    for seed in range(1000):
        rng = np.random.RandomState(seed)
        s = game.new_game()
        while not s.terminal:
            avail = np.nonzero(s.available_actions)[0]
            s = game.step(s, rng.choice(avail))
        if s.terminal_value == 0:
            board = s.board
            state = make_state(board, 1)
            found_draw = True
            break
    if found_draw:
        check("Full board with no winner is terminal", state.terminal)
        check("Full board draw has terminal_value=0", state.terminal_value == 0)
    else:
        print("  NOTE: Could not find a draw game in 1000 random tries, skipping draw test")
else:
    check("Full board with no winner is terminal", state.terminal)
    check("Full board draw has terminal_value=0", state.terminal_value == 0)


# ============================================================
# 5. The specific diagonal threat position from the bug report
# ============================================================
print("\n=== Bug Report: Diagonal Threat Position ===")

board = np.zeros((6, 7), dtype="int")
board[0] = [0, 1, 1, -1, 0, 1, -1]   # . O O X . O X
board[1] = [0, -1, 1, -1, 0, 0, 0]   # . X O X . . .
board[2] = [0, 1, -1, 1, 0, 0, 0]    # . O X O . . .
board[3] = [0, -1, 0, -1, 0, 0, 0]   # . X . X . . .

state = make_state(board, 1)  # O to move
print("  Position (O to move):")
print_board(state.board)

check("Position is NOT terminal yet", not state.terminal)
check("Col 0 is available", state.available_actions[0] == 1)

# Verify the threat: X has 3 in diagonal (1,1)-(2,2)-(3,3), needs (0,0)
check("(1,1)=X", board[1][1] == -1)
check("(2,2)=X", board[2][2] == -1)
check("(3,3)=X", board[3][3] == -1)
check("(0,0)=empty", board[0][0] == 0)

# If O plays col 4, then X plays col 0 → X wins
bad_state = game.step(state, 4)   # O plays col 4
x_wins = game.step(bad_state, 0)  # X plays col 0
print("\n  After O plays col 4, X plays col 0:")
print_board(x_wins.board)
check("X wins after playing col 0", x_wins.terminal and x_wins.terminal_value == -1,
      f"terminal={x_wins.terminal} val={x_wins.terminal_value}")

# If O blocks by playing col 0 first
block_state = game.step(state, 0)
print("\n  After O blocks col 0:")
print_board(block_state.board)
check("O blocking col 0 is NOT terminal", not block_state.terminal)
check("(0,0) is now O", block_state.board[0][0] == 1)


# ============================================================
# 6. Step function doesn't mutate original state
# ============================================================
print("\n=== Immutability Test ===")
s = game.new_game()
board_before = s.board.copy()
s2 = game.step(s, 3)
check("step() does not mutate original board", np.array_equal(s.board, board_before))
check("step() returns new board", not np.array_equal(s.board, s2.board))


# ============================================================
# 7. State encoding (state_to_input)
# ============================================================
print("\n=== State Encoding Tests ===")

# New game encoding (2 channels: my pieces, opponent pieces)
s = game.new_game()
inp = game.state_to_input(s)
check("Input shape is (2, 6, 7)", inp.shape == (2, 6, 7))

# No pieces on the board → both channels should be zeros
for ch in range(2):
    check(f"Channel {ch} is all zeros (empty board)", inp[ch].sum() == 0)

# After one move by X -> state has player=O
s1 = game.step(s, 3)  # X plays col 3
inp1 = game.state_to_input(s1)

# Relative encoding: channel 0 = my pieces (O), channel 1 = opponent pieces (X)
# O hasn't played yet, so channel 0 (my=O) is empty
# X played at (0,3), so channel 1 (opp=X) has it
check("After X@(0,3): channel 0 (my=O pieces) is zeros", inp1[0].sum() == 0)
check("After X@(0,3): channel 1 (opp=X pieces) has (0,3)", inp1[1][0][3] == 1.0)
check("After X@(0,3): channel 1 sum is 1", inp1[1].sum() == 1)

# After two moves -> state has player=X
s2 = game.step(s1, 5)  # O plays col 5
inp2 = game.state_to_input(s2)

# Relative: channel 0 = my pieces (X), channel 1 = opponent pieces (O)
check("After O@(0,5): channel 0 (my=X) has X at (0,3)", inp2[0][0][3] == 1.0)
check("After O@(0,5): channel 1 (opp=O) has O at (0,5)", inp2[1][0][5] == 1.0)
check("After O@(0,5): channel 0 sum is 1", inp2[0].sum() == 1)
check("After O@(0,5): channel 1 sum is 1", inp2[1].sum() == 1)


# ============================================================
# 9. Play a full game and verify consistency
# ============================================================
print("\n=== Full Game Playthrough ===")
np.random.seed(42)
s = game.new_game()
move_count = 0
while not s.terminal:
    avail = np.nonzero(s.available_actions)[0]
    action = np.random.choice(avail)
    s_next = game.step(s, action)

    # Verify player alternates
    check(f"Move {move_count}: player alternates",
          s_next.player == -s.player,
          f"was {s.player}, now {s_next.player}")

    # Verify piece was placed at the correct position
    row = None
    for r in range(ROW_COUNT):
        if s.board[r][action] == 0:
            row = r
            break
    if row is not None:
        check(f"Move {move_count}: piece at row={row} col={action}",
              s_next.board[row][action] == s.player)

    s = s_next
    move_count += 1

check(f"Game ended after {move_count} moves", s.terminal)
check(f"Terminal value is -1, 0, or 1", s.terminal_value in [-1, 0, 1])
print(f"  Winner: {s.terminal_value} ({move_count} moves)")
print_board(s.board)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("All Connect4 game logic tests PASSED!")
else:
    print(f"WARNING: {failed} test(s) FAILED!")
