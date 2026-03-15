import argparse

import torch

from network import AlphaZeroNet
from game_configs import GAME_CONFIGS
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
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move (default: per-game config)")
    parser.add_argument("--games", type=int, default=None,
                        help="Self-play games per iteration (default: per-game config)")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of training iterations (default: per-game config)")
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

    # Use per-game defaults when CLI args not specified
    num_iterations = args.iterations or game_cfg.get("default_iterations", 10)
    num_games = args.games or game_cfg.get("default_games", 32)
    num_simulations = args.simulations or game_cfg.get("default_simulations", 50)

    # Input channels: use game-specific value if available, else default formula
    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1))

    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=res_blocks,
        num_filters=filters,
        value_head_channels=game_cfg.get("value_head_channels", 2),
        value_head_fc_size=game_cfg.get("value_head_fc_size", 64),
        policy_head_channels=game_cfg.get("policy_head_channels", 2),
        ownership_channels=game_cfg.get("ownership_channels", 0),
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

    print(f"Config: {args.game} | iters={num_iterations} games={num_games} "
          f"sims={num_simulations}")

    config = {
        "num_simulations": num_simulations,
        "games_per_iteration": num_games,
        "checkpoint_dir": checkpoint_dir,
        "game_name": args.game,
        "device": device,
        "max_train_steps": game_cfg.get("max_train_steps", 6400),
        "target_epochs": game_cfg.get("target_epochs", 4),
        "buffer_size": game_cfg.get("buffer_size", 100000),
        "lr": game_cfg.get("lr", 0.01),
        "batch_size": game_cfg.get("batch_size", 64),
        "selects_per_round": game_cfg.get("selects_per_round", 1),
        "vl_value": game_cfg.get("vl_value", 0.0),
        "value_loss_weight": game_cfg.get("value_loss_weight", 1.0),
        "temp_threshold": game_cfg.get("temp_threshold", 15),
        "c_puct": game_cfg.get("c_puct", 1.5),
        "dirichlet_alpha": game_cfg.get("dirichlet_alpha", 1.0),
        "tree_reuse": game_cfg.get("tree_reuse", True),
        "resign_threshold": game_cfg.get("resign_threshold", -1.0),
        "resign_min_moves": game_cfg.get("resign_min_moves", 99),
        "resign_check_prob": game_cfg.get("resign_check_prob", 0.0),
        "ownership_loss_weight": game_cfg.get("ownership_loss_weight", 0.0),
    }

    trainer = Trainer(game, net, config)
    trainer.run(num_iterations=num_iterations)


if __name__ == "__main__":
    main()
