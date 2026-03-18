import argparse

import torch

from game_configs import GAME_CONFIGS
from training import Trainer
from utils import GAMES, load_game, make_net


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

    # Use per-game defaults when CLI args not specified
    num_iterations = args.iterations or game_cfg.get("default_iterations", 10)
    num_games = args.games or game_cfg.get("default_games", 32)
    num_simulations = args.simulations or game_cfg.get("default_simulations", 50)

    net = make_net(game, args.game)

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

    random_opening = game_cfg.get('random_opening_moves', 0)
    print(f"Config: {args.game} | iters={num_iterations} games={num_games} "
          f"sims={num_simulations}")
    if random_opening > 0:
        print(f"  Forced openings: 0-{random_opening} random moves per game")

    config = {
        **game_cfg,
        "num_simulations": num_simulations,
        "games_per_iteration": num_games,
        "checkpoint_dir": checkpoint_dir,
        "game_name": args.game,
        "device": device,
    }

    trainer = Trainer(game, net, config)
    trainer.run(num_iterations=num_iterations)


if __name__ == "__main__":
    main()
