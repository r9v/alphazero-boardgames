"""Tournament: battle all checkpoints in elimination style.

Usage:
    python battle/tournament.py --game connect4 [--sims 200] [--games 50] [--parallel 10]

Loads all .pt checkpoints from checkpoints/<game>/, seeds a single-elimination
bracket, and plays matches. Winner advances. Prints bracket results.
"""
import argparse
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_game, make_net
from mcts import MCTS, Node, add_dirichlet_noise


def _batched_mcts_move(game, net, indices, states, sims, c_puct):
    """Run MCTS for all games in `indices` with batched neural net inference.

    Uses virtual-loss multi-select to collect leaf nodes across all games,
    then evaluates them in a single batch_predict call.
    """

    mcts_instances = {i: MCTS(game, net, c_puct=c_puct) for i in indices}
    # Create root nodes
    roots = {}
    root_inputs = []
    root_idx_map = []
    for i in indices:
        root = Node(None, states[i], game)  # net=None, deferred
        roots[i] = root
        root_inputs.append(game.state_to_input(states[i]))
        root_idx_map.append(i)

    # Batch evaluate roots
    if root_inputs:
        values, policies = net.batch_predict(root_inputs)
        for j, i in enumerate(root_idx_map):
            roots[i].resolve(values[j], policies[j])
            # Dirichlet noise so each game plays out differently
            roots[i].P = add_dirichlet_noise(
                roots[i].P, 0.03, 0.25, roots[i].available_actions_mask)

    # Run simulations with batched leaf evaluation
    for _sim in range(sims):
        # Collect leaves across all games using virtual loss
        leaves = []  # (game_idx, leaf_node, path)
        leaf_inputs = []
        for i in indices:
            leaf, path = mcts_instances[i].search_expand_vl(roots[i])
            if leaf is not None:
                leaves.append((i, leaf, path))
                leaf_inputs.append(game.state_to_input(leaf.state))

        if not leaf_inputs:
            continue

        # Single batched inference for all leaves
        values, policies = net.batch_predict(leaf_inputs)

        # Resolve and backprop
        for j, (i, leaf, path) in enumerate(leaves):
            leaf.resolve(values[j], policies[j])
            mcts_instances[i].search_backup_vl(leaf, path)

    # Extract actions from visit counts
    actions = {}
    for i in indices:
        root = roots[i]
        pi = np.zeros(len(root.available_actions_mask), dtype=np.float64)
        for child_action in root.available_actions:
            child = root.children[child_action]
            if child is not None:
                pi[child_action] = child.n
        if pi.sum() > 0:
            pi /= pi.sum()
        actions[i] = np.argmax(pi)

    return actions


def play_match(game, net1, net2, num_games, sims, c_puct, parallel=10, device='cuda'):
    """Play num_games between net1 and net2 with batched parallelism.

    Runs `parallel` games simultaneously, batching MCTS leaf evaluations
    across all games into single neural net inference calls.
    Returns (net1_wins, net2_wins, draws).
    """
    wins1, wins2, draws = 0, 0, 0
    games_done = 0

    while games_done < num_games:
        batch_size = min(parallel, num_games - games_done)

        # Initialize parallel games
        states = [game.new_game() for _ in range(batch_size)]
        # Alternate who goes first: even=net1 first, odd=net2 first
        first_is_1 = [(games_done + i) % 2 == 0 for i in range(batch_size)]
        move_counts = [0] * batch_size
        active = list(range(batch_size))

        while active:
            # Group by which network should move
            net1_indices = []
            net2_indices = []
            for i in active:
                is_first_player_turn = move_counts[i] % 2 == 0
                if (is_first_player_turn and first_is_1[i]) or \
                   (not is_first_player_turn and not first_is_1[i]):
                    net1_indices.append(i)
                else:
                    net2_indices.append(i)

            # Batch MCTS with batched neural net inference
            for indices, net in [(net1_indices, net1), (net2_indices, net2)]:
                if not indices:
                    continue

                actions = _batched_mcts_move(game, net, indices, states, sims, c_puct)
                for i in indices:
                    states[i] = game.step(states[i], actions[i])
                    move_counts[i] += 1

            # Check for terminal states
            new_active = []
            for i in active:
                if states[i].terminal:
                    tv = states[i].terminal_value
                    if tv == 0:
                        draws += 1
                    elif (tv == -1 and first_is_1[i]) or (tv == 1 and not first_is_1[i]):
                        wins1 += 1
                    else:
                        wins2 += 1
                else:
                    new_active.append(i)
            active = new_active

        games_done += batch_size
        ts = time.strftime("%H:%M:%S")
        print(f"      [{ts}] games {games_done}/{num_games}: {wins1}-{wins2} (d={draws})")

    return wins1, wins2, draws


def load_checkpoint(game, game_name, path, device='cuda'):
    """Create a fresh network and load checkpoint."""
    net = make_net(game, game_name)
    if not net.load(path):
        raise RuntimeError(f"Failed to load checkpoint: {path}")
    net.eval()
    net.to(device)
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
    parser.add_argument("--parallel", type=int, default=10,
                        help="Games to run in parallel per match (default 10)")
    args = parser.parse_args()

    game = load_game(args.game)
    checkpoints = get_checkpoints(args.game)
    n = len(checkpoints)
    print(f"Found {n} checkpoints for {args.game}")
    print(f"Settings: {args.sims} sims, {args.games} games/match, "
          f"c_puct={args.c_puct}, parallel={args.parallel}")

    if n < 2:
        print("Need at least 2 checkpoints for a tournament.")
        return

    contestants = list(checkpoints)
    round_num = 1

    while len(contestants) > 1:
        print(f"\n{'='*60}")
        print(f"ROUND {round_num} — {len(contestants)} contestants")
        print(f"{'='*60}")

        next_round = []

        for i in range(0, len(contestants), 2):
            if i + 1 >= len(contestants):
                print(f"\n  {short_name(contestants[i])} gets a bye")
                next_round.append(contestants[i])
                continue

            path1, path2 = contestants[i], contestants[i + 1]
            name1, name2 = short_name(path1), short_name(path2)
            print(f"\n  Match: {name1} vs {name2}")

            t0 = time.time()
            net1 = load_checkpoint(game, args.game, path1)
            net2 = load_checkpoint(game, args.game, path2)

            w1, w2, d = play_match(game, net1, net2, args.games, args.sims,
                                   args.c_puct, parallel=args.parallel)
            elapsed = time.time() - t0

            print(f"    {name1}: {w1}W  |  {name2}: {w2}W  |  draws: {d}  ({elapsed:.1f}s)")

            # Tie-break: rematch with 2x games
            if w1 == w2:
                rematch_games = args.games * 2
                print(f"    Tied! Rematch with {rematch_games} games...")
                t0 = time.time()
                w1, w2, d = play_match(game, net1, net2, rematch_games, args.sims,
                                       args.c_puct, parallel=args.parallel)
                elapsed = time.time() - t0
                print(f"    {name1}: {w1}W  |  {name2}: {w2}W  |  draws: {d}  ({elapsed:.1f}s)")

            # Free GPU memory
            del net1, net2
            torch.cuda.empty_cache()

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
