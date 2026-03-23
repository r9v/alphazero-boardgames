"""Tournament: battle all checkpoints in elimination style.

Usage:
    python battle/tournament.py --game connect4 [--sims 200] [--games 50] [--c-puct 2.5]

Loads all .pt checkpoints from checkpoints/<game>/, seeds a single-elimination
bracket, and plays matches. Winner advances. Prints bracket results.
"""
import argparse
import os
import sys
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_game, make_net
from mcts import MCTS


def play_match(game, net1, net2, mcts1, mcts2, num_games, sims):
    """Play num_games between net1 and net2. Returns (net1_wins, net2_wins, draws)."""
    wins1, wins2, draws = 0, 0, 0

    for g in range(num_games):
        # Alternate who goes first
        if g % 2 == 0:
            first_net, second_net = net1, net2
            first_mcts, second_mcts = mcts1, mcts2
            first_is_1 = True
        else:
            first_net, second_net = net2, net1
            first_mcts, second_mcts = mcts2, mcts1
            first_is_1 = False

        state = game.new_game()
        move_count = 0

        while not state.terminal:
            if move_count % 2 == 0:
                mcts_player = first_mcts
            else:
                mcts_player = second_mcts

            pi = mcts_player.get_policy(sims, state, add_dirichlet=False)
            action = np.argmax(pi)
            state = game.step(state, action)
            move_count += 1

        tv = state.terminal_value
        if tv == 0:
            draws += 1
        elif tv == -1:  # player -1 (first mover) won
            if first_is_1:
                wins1 += 1
            else:
                wins2 += 1
        else:  # player +1 (second mover) won
            if first_is_1:
                wins2 += 1
            else:
                wins1 += 1

    return wins1, wins2, draws


def load_checkpoint(game, game_name, path):
    """Create a fresh network and load checkpoint."""
    net = make_net(game, game_name)
    if not net.load(path):
        raise RuntimeError(f"Failed to load checkpoint: {path}")
    net.eval()
    return net


def get_checkpoints(game_name):
    """Get all .pt checkpoint paths sorted by name (chronological)."""
    ckpt_dir = os.path.join("checkpoints", game_name)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint directory: {ckpt_dir}")

    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
    if not files:
        raise FileNotFoundError(f"No .pt files in {ckpt_dir}")

    return [os.path.join(ckpt_dir, f) for f in files]


def short_name(path):
    """Extract short name from checkpoint path."""
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(description="Checkpoint elimination tournament")
    parser.add_argument("--game", default="connect4")
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--games", type=int, default=50, help="Games per match")
    parser.add_argument("--c-puct", type=float, default=2.5)
    args = parser.parse_args()

    game = load_game(args.game)
    checkpoints = get_checkpoints(args.game)
    n = len(checkpoints)
    print(f"Found {n} checkpoints for {args.game}")
    print(f"Settings: {args.sims} sims, {args.games} games/match, c_puct={args.c_puct}")

    if n < 2:
        print("Need at least 2 checkpoints for a tournament.")
        return

    # Seed the bracket — pair adjacent checkpoints (chronological neighbors)
    # If odd number, last checkpoint gets a bye
    contestants = list(checkpoints)
    round_num = 1

    while len(contestants) > 1:
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} — {len(contestants)} contestants")
        print(f"{'='*60}")

        next_round = []

        for i in range(0, len(contestants), 2):
            if i + 1 >= len(contestants):
                # Bye — odd one out advances
                print(f"\n  {short_name(contestants[i])} gets a bye")
                next_round.append(contestants[i])
                continue

            path1, path2 = contestants[i], contestants[i + 1]
            name1, name2 = short_name(path1), short_name(path2)
            print(f"\n  Match: {name1} vs {name2}")

            t0 = time.time()
            net1 = load_checkpoint(game, args.game, path1)
            net2 = load_checkpoint(game, args.game, path2)
            mcts1 = MCTS(game, net1, c_puct=args.c_puct)
            mcts2 = MCTS(game, net2, c_puct=args.c_puct)

            w1, w2, d = play_match(game, net1, net2, mcts1, mcts2, args.games, args.sims)
            elapsed = time.time() - t0

            print(f"    {name1}: {w1}W  |  {name2}: {w2}W  |  draws: {d}  ({elapsed:.1f}s)")

            if w1 >= w2:
                winner = path1
                print(f"    Winner: {name1}")
            else:
                winner = path2
                print(f"    Winner: {name2}")

            next_round.append(winner)

        contestants = next_round
        round_num += 1

    print(f"\n{'='*60}")
    print(f"CHAMPION: {short_name(contestants[0])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
