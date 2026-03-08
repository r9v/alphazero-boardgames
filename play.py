import argparse
import numpy as np

from network import AlphaZeroNet
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
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--filters", type=int, default=256)
    parser.add_argument("--res-blocks", type=int, default=2)
    parser.add_argument("--human-first", action="store_true",
                        help="Human plays first (as X)")
    args = parser.parse_args()

    # Santorini launches the pygame GUI instead of terminal play
    if args.game == "santorini":
        from games.santorini.gui import GUI
        ai_player = 1 if args.human_first else -1
        GUI(ai_player=ai_player, simulations=args.simulations,
            filters=args.filters, res_blocks=args.res_blocks)
        return

    # Load game
    if args.game == "tictactoe":
        from games.tictactoe import TTTGame
        game = TTTGame()
    elif args.game == "connect4":
        from games.connect4 import Connect4Game
        game = Connect4Game()

    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1) + 2)
    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=args.res_blocks,
        num_filters=args.filters,
    )

    checkpoint_dir = f"checkpoints/{args.game}"
    if net.load_latest(checkpoint_dir):
        print("Loaded trained model.")
    else:
        print("No checkpoint found, using untrained network.")

    mcts = MCTS(game, net)
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
            pi = mcts.get_policy(args.simulations, state)
            move = np.argmax(pi)
            print(f"AI plays: {move}")
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
