import argparse

from network import AlphaZeroNet
from training import Trainer


GAMES = {
    "tictactoe": "games.tictactoe:TTTGame",
    "connect4": "games.connect4:Connect4Game",
    "santorini": "games.santorini:SantoriniGame",
}


def load_game(name):
    if name not in GAMES:
        raise ValueError(f"Unknown game '{name}'. Choose from: {list(GAMES.keys())}")
    module_path, class_name = GAMES[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def main():
    parser = argparse.ArgumentParser(description="AlphaZero self-play training")
    parser.add_argument("--game", type=str, default="tictactoe",
                        choices=list(GAMES.keys()))
    parser.add_argument("--simulations", type=int, default=50,
                        help="MCTS simulations per move")
    parser.add_argument("--games", type=int, default=2,
                        help="Self-play games per iteration")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of training iterations")
    parser.add_argument("--filters", type=int, default=256,
                        help="Number of conv filters")
    parser.add_argument("--res-blocks", type=int, default=2,
                        help="Number of residual blocks")
    args = parser.parse_args()

    game = load_game(args.game)

    # Input channels: use game-specific value if available, else default formula
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

    config = {
        "num_simulations": args.simulations,
        "games_per_iteration": args.games,
        "checkpoint_dir": checkpoint_dir,
        "game_name": args.game,
    }

    trainer = Trainer(game, net, config)
    trainer.run(num_iterations=args.iterations)


if __name__ == "__main__":
    main()
