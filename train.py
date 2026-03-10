import argparse

import torch

from network import AlphaZeroNet, GAME_CONFIGS
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
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto' (default)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    game = load_game(args.game)

    game_cfg = GAME_CONFIGS.get(args.game, {})
    filters = game_cfg.get("num_filters", 256)
    res_blocks = game_cfg.get("num_res_blocks", 2)

    # Input channels: use game-specific value if available, else default formula
    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1))

    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=res_blocks,
        num_filters=filters,
    )

    checkpoint_dir = f"checkpoints/{args.game}"
    net.to(device)

    # Resume from latest checkpoint if available
    loaded_path = net.load_latest(checkpoint_dir)
    if loaded_path:
        print(f"Resumed from: {loaded_path}")
    else:
        print("No checkpoint found, starting from scratch.")

    if device == "cuda":
        net.compile_for_inference()

    config = {
        "num_simulations": args.simulations,
        "games_per_iteration": args.games,
        "checkpoint_dir": checkpoint_dir,
        "game_name": args.game,
        "device": device,
        "max_train_steps": game_cfg.get("max_train_steps", 1000),
        "selects_per_round": game_cfg.get("selects_per_round", 1),
        "vl_value": game_cfg.get("vl_value", 0.0),
        "value_loss_weight": game_cfg.get("value_loss_weight", 1.0),
    }

    trainer = Trainer(game, net, config)
    trainer.run(num_iterations=args.iterations)


if __name__ == "__main__":
    main()
