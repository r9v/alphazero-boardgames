import argparse
import numpy as np

from network import AlphaZeroNet
from game_configs import GAME_CONFIGS
from mcts import MCTS


def print_board(board, game_name):
    if game_name == "tictactoe":
        symbols = {0: ".", -1: "X", 1: "O"}
        print()
        print("  Board:        Move indices:")
        print()
        for r in range(3):
            pieces = " ".join(symbols[board[r][c]] for c in range(3))
            indices = " ".join(str(r * 3 + c) for c in range(3))
            print(f"    {pieces}           {indices}")
        print()

    elif game_name == "connect4":
        symbols = {0: ".", -1: "X", 1: "O"}
        print()
        for r in range(5, -1, -1):  # top to bottom
            row = " ".join(symbols[board[r][c]] for c in range(7))
            print(f"    {row}")
        print("    " + " ".join(str(c) for c in range(7)))
        print()

    elif game_name == "santorini":
        print()
        for r in range(5):
            cells = []
            for c in range(5):
                cells.append(str(board[r][c]))
            print(f"    {' '.join(cells)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Play against AlphaZero")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=["tictactoe", "connect4", "santorini"])
    parser.add_argument("--simulations", type=int, default=None,
                        help="Override play-time simulations (default: from game config)")
    parser.add_argument("--human-first", action="store_true",
                        help="Human plays first (as X)")
    args = parser.parse_args()

    net_cfg = GAME_CONFIGS.get(args.game, {})
    filters = net_cfg.get("num_filters", 256)
    res_blocks = net_cfg.get("num_res_blocks", 2)

    # Resolve play-time config from game config
    play_sims = args.simulations or net_cfg.get("play_simulations", 100)
    play_c_puct = net_cfg.get("play_c_puct", 1.5)

    # Santorini launches the pygame GUI instead of terminal play
    if args.game == "santorini":
        from games.santorini.gui import GUI
        ai_player = 1 if args.human_first else -1
        GUI(ai_player=ai_player, simulations=play_sims,
            filters=filters, res_blocks=res_blocks, c_puct=play_c_puct,
            value_head_channels=net_cfg.get("value_head_channels", 2),
            value_head_fc_size=net_cfg.get("value_head_fc_size", 64))
        return

    from train import load_game
    game = load_game(args.game)

    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1))
    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=res_blocks,
        num_filters=filters,
        value_head_channels=net_cfg.get("value_head_channels", 2),
        value_head_fc_size=net_cfg.get("value_head_fc_size", 64),
    )

    checkpoint_dir = f"checkpoints/{args.game}"
    loaded_path = net.load_latest(checkpoint_dir)
    if loaded_path:
        print(f"Loaded model: {loaded_path}")
    else:
        print("No checkpoint found, using untrained network.")

    mcts_mod = MCTS.__module__
    mcts_label = "C/Cython" if "c_mcts" in mcts_mod else "Python"
    game_mod = type(game).__module__
    game_label = "C/Cython" if "c_game" in game_mod else "Python"
    print(f"Config: sims={play_sims} c_puct={play_c_puct}")
    print(f"  MCTS backend: {mcts_label} ({mcts_mod})")
    print(f"  Game backend: {game_label} ({game_mod})")
    mcts = MCTS(game, net, c_puct=play_c_puct)
    human_player = -1 if args.human_first else 1  # -1 goes first

    state = game.new_game()
    print(f"\nYou are {'X' if human_player == -1 else 'O'}. AI is {'O' if human_player == -1 else 'X'}.")

    while not state.terminal:
        print_board(state.board, args.game)

        if state.player == human_player:
            available = np.nonzero(state.available_actions)[0]
            while True:
                try:
                    move = int(input(f"Your move {[int(a) for a in available]}: "))
                    if move in available:
                        break
                    print("Invalid move.")
                except (ValueError, EOFError):
                    print("Enter a number.")
            state = game.step(state, move)
        else:
            print("AI thinking...")
            pi = mcts.get_policy(play_sims, state)
            move = np.argmax(pi)

            # Debug: show MCTS stats per action
            root = mcts.last_root
            print(f"\n  {'Col':>3}  {'N':>6}  {'Q':>7}  {'P':>7}  {'pi':>7}")
            print(f"  {'---':>3}  {'---':>6}  {'---':>7}  {'---':>7}  {'---':>7}")
            for a in range(len(pi)):
                child = root.children.get(a) if isinstance(root.children, dict) else root.children[a]
                n = child.n if child else 0
                q = f"{child.Q:+.3f}" if child else "  -   "
                p = f"{root.P[a]:.3f}" if root.P[a] > 0.001 else "  .   "
                pi_str = f"{pi[a]:.3f}" if pi[a] > 0.001 else "  .   "
                marker = " <--" if a == move else ""
                print(f"  {a:>3}  {n:>6}  {q:>7}  {p:>7}  {pi_str:>7}{marker}")
            print(f"\n  Root N={root.n}  V={root.nnet_value:+.3f}")
            print(f"AI plays: {move}\n")
            state = game.step(state, move)

    print_board(state.board, args.game)

    if state.terminal_value == 0:
        print("Draw!")
    elif state.terminal_value == human_player:
        print("You win!")
    else:
        print("AI wins!")


if __name__ == "__main__":
    main()
