"""Debug script to analyze a specific Connect4 position where AI missed a diagonal threat."""
import numpy as np
from games.connect4 import Connect4Game, GameState
from network import AlphaZeroNet, GAME_CONFIGS
from mcts import MCTS


def build_position():
    """Reconstruct the board position from the bug report.

    Display (row 5=top, row 0=bottom):
        . . . . . . .     r=5
        . . . . . . .     r=4
        . X . X . . .     r=3
        . O X O . . .     r=2
        . X O X . . .     r=1
        . O O X O O X     r=0
        0 1 2 3 4 5 6

    It's X's turn (player=-1). X can win immediately by playing column 0
    (completing diagonal (0,0)-(1,1)-(2,2)-(3,3)).
    """
    game = Connect4Game()

    # Position BEFORE AI (O) played column 4
    # Display:
    #   . . . . . . .     r=5
    #   . . . . . . .     r=4
    #   . X . X . . .     r=3
    #   . O X O . . .     r=2
    #   . X O X . . .     r=1
    #   . O O X . O X     r=0  (col 4 empty - AI hasn't played yet)
    #   0 1 2 3 4 5 6
    # O's turn (player=1). O must play col 0 to block diagonal threat.
    board = np.zeros((6, 7), dtype="int")
    board[0] = [0, 1, 1, -1, 0, 1, -1]   # . O O X . O X
    board[1] = [0, -1, 1, -1, 0, 0, 0]   # . X O X . . .
    board[2] = [0, 1, -1, 1, 0, 0, 0]    # . O X O . . .
    board[3] = [0, -1, 0, -1, 0, 0, 0]   # . X . X . . .
    # rows 4,5 = empty

    state = GameState(None, board, player=1)  # O to move
    return game, state


def print_board(board):
    symbols = {0: ".", -1: "X", 1: "O"}
    for r in range(5, -1, -1):
        row = " ".join(symbols[board[r][c]] for c in range(7))
        print(f"    {row}")
    print("    " + " ".join(str(c) for c in range(7)))


def main():
    game, state = build_position()

    print("=== Reconstructed Position ===")
    print_board(state.board)
    print(f"Player to move: {state.player} ({'X' if state.player == -1 else 'O'})")
    print(f"Terminal: {state.terminal}")
    print()

    # Verify the threat: if O doesn't block, X plays column 0 next
    # First show what happens if O plays col 4 (the bad move)
    bad_state = game.step(state, 4)  # O plays col 4
    threat_state = game.step(bad_state, 0)  # X plays col 0
    print("=== After O plays col 4, then X plays col 0 ===")
    print_board(threat_state.board)
    print(f"Terminal: {threat_state.terminal}, Value: {threat_state.terminal_value}")

    # Check the diagonal manually
    b = threat_state.board
    print(f"\nDiagonal check (0,0)-(1,1)-(2,2)-(3,3):")
    for r in range(4):
        symbols = {0: ".", -1: "X", 1: "O"}
        print(f"  ({r},{r}) = {b[r][r]} ({symbols[b[r][r]]})")
    all_x = all(b[r][r] == -1 for r in range(4))
    print(f"  All X? {all_x}")

    if not threat_state.terminal:
        print("\n*** BUG: Win detection FAILED! ***")
        print(f"  _check_player_won(-1) = {threat_state._check_player_won(-1)}")
        return
    else:
        print("\nWin detection OK - X wins with diagonal.")

    # Load the network
    cfg = GAME_CONFIGS["connect4"]
    net = AlphaZeroNet(
        input_channels=2 * (game.num_history_states + 1),
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=cfg["num_res_blocks"],
        num_filters=cfg["num_filters"],
    )
    loaded = net.load_latest("checkpoints/connect4")
    if loaded:
        print(f"\nLoaded model: {loaded}")
    else:
        print("\nNo checkpoint found!")
        return

    # Raw network evaluation
    print("\n=== Raw Neural Network Evaluation ===")
    state_input = game.state_to_input(state)
    value, policy = net.predict(state_input)
    print(f"Value (nnet_value): {value:+.4f}")
    print(f"  (positive = O wins, negative = X wins)")
    print(f"  Player to move: {state.player} ({'X' if state.player == -1 else 'O'})")
    print(f"\nPolicy priors:")
    for col in range(7):
        avail = "Y" if state.available_actions[col] else "N"
        print(f"  Col {col}: P={policy[col]:.4f}  avail={avail}")

    # MCTS analysis at various sim counts
    print("\n=== MCTS Analysis ===")
    for sims in [100, 200, 500, 1000]:
        mcts = MCTS(game, net)
        pi = mcts.get_policy(sims, state)
        root = mcts.last_root

        best = np.argmax(pi)
        print(f"\n--- {sims} simulations ---")
        print(f"  Root N={root.n}  nnet_value={root.nnet_value:+.4f}  Q={root.Q:+.4f}")
        print(f"  {'Col':>3}  {'N':>6}  {'Q':>7}  {'P':>7}  {'pi':>7}")
        print(f"  {'---':>3}  {'---':>6}  {'---':>7}  {'---':>7}  {'---':>7}")
        for a in range(7):
            child = root.children[a] if isinstance(root.children, list) else root.children.get(a)
            if child is not None:
                n = child.n
                q = f"{child.Q:+.4f}"
                p = f"{root.P[a]:.4f}"
                pi_str = f"{pi[a]:.4f}"
                marker = ""
                if a == best:
                    marker += " <-- BEST"
                if a == 0:
                    marker += " ** BLOCK"
                print(f"  {a:>3}  {n:>6}  {q:>7}  {p:>7}  {pi_str:>7}{marker}")
            elif state.available_actions[a]:
                extra = " ** BLOCK (unexplored!)" if a == 0 else ""
                print(f"  {a:>3}  {0:>6}  {'  -   ':>7}  {root.P[a]:.4f}  {'  .   ':>7}{extra}")

    # Check encoding
    print("\n=== Encoding Check ===")
    inp = game.state_to_input(state)
    print(f"Input shape: {inp.shape}")
    print(f"History channels (0-3) all zeros (no prev_state):")
    for ch in range(4):
        total = inp[ch].sum()
        print(f"  Channel {ch}: sum={total:.0f}")
    c = 2 * game.num_history_states  # index of current board channels
    print(f"Channel {c} (current X pieces, board==-1):")
    for r in range(5, -1, -1):
        print(f"  row {r}: {inp[c][r].astype(int).tolist()}")
    print(f"Channel {c+1} (current O pieces, board==1):")
    for r in range(5, -1, -1):
        print(f"  row {r}: {inp[c+1][r].astype(int).tolist()}")
    print(f"Channel {c+2} (player -1 indicator): all={inp[c+2].mean():.0f}")
    print(f"Channel {c+3} (player +1 indicator): all={inp[c+3].mean():.0f}")

    # Now test with realistic history by replaying actual moves
    print("\n\n=== WITH HISTORY (replayed game) ===")
    # Replay a plausible move sequence that produces this board
    # Col 1: O@0, X@1, O@2, X@3
    # Col 2: O@0, O@1, X@2
    # Col 3: X@0, X@1, O@2, X@3
    # Col 4: (empty)
    # Col 5: O@0
    # Col 6: X@0
    moves = [
        3,  # X@(0,3)
        1,  # O@(0,1)
        6,  # X@(0,6)
        2,  # O@(0,2)
        1,  # X@(1,1)
        2,  # O@(1,2)
        3,  # X@(1,3)
        5,  # O@(0,5)
        2,  # X@(2,2)
        1,  # O@(2,1)
        3,  # X@(2,3) -- wait, (2,3) should be O
    ]
    # Col 3 needs: X@0, X@1, O@2, X@3. Let me fix the sequence.
    # X plays odd indices (0,2,4,...), O plays even indices (1,3,5,...)
    # Using 0-indexed move number: move 0=X, move 1=O, move 2=X, ...
    moves = [
        3,  # move 0: X@(0,3)
        1,  # move 1: O@(0,1)
        6,  # move 2: X@(0,6)
        2,  # move 3: O@(0,2)
        1,  # move 4: X@(1,1)
        2,  # move 5: O@(1,2)
        3,  # move 6: X@(1,3)  -- check: no win, just 2 vertical at col3
        5,  # move 7: O@(0,5)
        2,  # move 8: X@(2,2)  -- check diagonal: (2,2)X,(1,1)X = 2, no win
        1,  # move 9: O@(2,1)
        1,  # move 10: X@(3,1)  -- check: (3,1)X,(2,2)X,(1,3)X = 3 diagonal! need (0,4) for win but (0,4) empty, not X. (4,0) out of range. no win
        3,  # move 11: O@(2,3)
        3,  # move 12: X@(3,3)  -- check diagonal: (0,0)=.,(1,1)=X,(2,2)=X,(3,3)=X = 3 in row. (0,0) empty. Not 4 yet.
    ]
    # That's 13 moves: 7 X, 6 O = total 13 pieces. Player=O next.
    # Board should be:
    # Col 0: empty
    # Col 1: O(0), X(1), O(2), X(3) -- 4 pieces
    # Col 2: O(0), O(1), X(2) -- 3 pieces
    # Col 3: X(0), X(1), O(2), X(3) -- 4 pieces
    # Col 5: O(0) -- 1 piece
    # Col 6: X(0) -- 1 piece
    # Total: 13 correct. X=7, O=6.

    s = game.new_game()
    for i, m in enumerate(moves):
        s = game.step(s, m)
        if s.terminal:
            print(f"  Game ended at move {i} (col {m})! terminal_value={s.terminal_value}")
            print_board(s.board)
            return

    print("Board after replay:")
    print_board(s.board)
    print(f"Player: {s.player} ({'X' if s.player == -1 else 'O'})")

    # Check it matches
    if np.array_equal(s.board, state.board):
        print("Board MATCHES target!")
    else:
        print("Board MISMATCH!")
        print("Expected:")
        print_board(state.board)
        return

    # Now evaluate with history
    inp2 = game.state_to_input(s)
    print(f"\nHistory channels with replay:")
    for ch in range(4):
        total = inp2[ch].sum()
        print(f"  Channel {ch}: sum={total:.0f}")

    value2, policy2 = net.predict(inp2)
    print(f"\nWith history - Value: {value2:+.4f} (was {value:+.4f} without)")
    print(f"With history - Policy:")
    for col in range(7):
        diff = policy2[col] - policy[col]
        print(f"  Col {col}: P={policy2[col]:.4f} (delta={diff:+.4f})")

    # Run MCTS with history
    print(f"\n--- MCTS with history (100 sims) ---")
    mcts2 = MCTS(game, net)
    pi2 = mcts2.get_policy(100, s)
    root2 = mcts2.last_root
    best2 = np.argmax(pi2)
    print(f"  Root N={root2.n}  nnet_value={root2.nnet_value:+.4f}")
    for a in range(7):
        child = root2.children[a] if isinstance(root2.children, list) else root2.children.get(a)
        if child is not None:
            marker = " <-- BEST" if a == best2 else ""
            if a == 0:
                marker += " ** BLOCK"
            print(f"  Col {a}: N={child.n:>4}  Q={child.Q:+.4f}  P={root2.P[a]:.4f}  pi={pi2[a]:.4f}{marker}")
        elif s.available_actions[a]:
            extra = " ** BLOCK (unexplored!)" if a == 0 else ""
            print(f"  Col {a}: N=   0  Q=  -     P={root2.P[a]:.4f}  pi=  .   {extra}")


if __name__ == "__main__":
    main()
