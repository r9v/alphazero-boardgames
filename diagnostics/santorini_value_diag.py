"""Santorini value head diagnostic: overfitting, symmetry, and target noise analysis.

Tests:
1. Symmetry consistency: does the model give the same value for rotated positions?
2. Overfitting check: train on small set, test on held-out set
3. Target noise analysis: how noisy are value targets from random self-play?
"""
import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_game
from network import AlphaZeroNet
from game_configs import GAME_CONFIGS


def rotate_90(board_5x5):
    """Rotate a 5x5 array 90 degrees clockwise."""
    return np.rot90(board_5x5, k=-1)


def rotate_state_input(state_input, k=1):
    """Rotate all channels of a state_input by k*90 degrees clockwise."""
    rotated = np.array(state_input)
    for ch in range(rotated.shape[0]):
        rotated[ch] = np.rot90(rotated[ch], k=-k)
    return rotated


def reflect_state_input(state_input):
    """Reflect state_input horizontally (left-right flip)."""
    reflected = np.array(state_input)
    for ch in range(reflected.shape[0]):
        reflected[ch] = np.fliplr(reflected[ch])
    return reflected


def test_symmetry_consistency(game, net, n_games=20, n_sims=0):
    """Test if the network gives consistent values for rotated/reflected positions.

    If the network has learned meaningful features, it should give similar values
    for positions that are equivalent under rotation/reflection. If it hasn't
    (e.g. because training data lacks augmentation), values will differ wildly.
    """
    print("=" * 70)
    print("TEST 1: Symmetry Consistency")
    print("=" * 70)

    # Generate positions from random games
    positions = []
    for _ in range(n_games):
        state = game.new_game()
        game_positions = []
        while not state.terminal:
            game_positions.append(state)
            mask = state.available_actions
            actions = np.where(mask == 1)[0]
            action = actions[np.random.randint(len(actions))]
            state = game.step(state, action)
        # Take positions from middle of game (more interesting)
        if len(game_positions) > 4:
            for idx in range(2, len(game_positions) - 2, 3):
                positions.append(game_positions[idx])

    if not positions:
        print("No positions generated!")
        return

    print(f"Evaluating {len(positions)} positions under 4 rotations + 4 reflections...")

    value_diffs_rot = []  # max value diff across rotations
    value_diffs_all = []  # max value diff across all 8 symmetries

    for state in positions:
        inp = game.state_to_input(state)
        values = []

        # 4 rotations
        for k in range(4):
            rotated = rotate_state_input(inp, k)
            v, _ = net.predict(rotated)
            values.append(v)

        # 4 reflected + rotated
        reflected = reflect_state_input(inp)
        for k in range(4):
            rot_ref = rotate_state_input(reflected, k)
            v, _ = net.predict(rot_ref)
            values.append(v)

        values = np.array(values)
        rot_range = values[:4].max() - values[:4].min()
        all_range = values.max() - values.min()
        value_diffs_rot.append(rot_range)
        value_diffs_all.append(all_range)

    rot_diffs = np.array(value_diffs_rot)
    all_diffs = np.array(value_diffs_all)

    print(f"\nRotation-only value spread (should be ~0 if rotation-invariant):")
    print(f"  mean={rot_diffs.mean():.4f}  median={np.median(rot_diffs):.4f}  "
          f"max={rot_diffs.max():.4f}  >0.5: {(rot_diffs > 0.5).mean():.1%}")
    print(f"\nAll 8 symmetries value spread:")
    print(f"  mean={all_diffs.mean():.4f}  median={np.median(all_diffs):.4f}  "
          f"max={all_diffs.max():.4f}  >0.5: {(all_diffs > 0.5).mean():.1%}")

    if rot_diffs.mean() > 0.3:
        print("\n*** HIGH ROTATION VARIANCE: Network is NOT rotation-invariant!")
        print("    This wastes capacity learning separate patterns for each rotation.")
        print("    FIX: Add symmetry augmentation to training data.")
    elif rot_diffs.mean() > 0.1:
        print("\n** MODERATE rotation variance — some invariance learned, but incomplete.")
    else:
        print("\n   Good rotation invariance.")


def test_target_noise(game, n_games=200):
    """Analyze how noisy value targets are from self-play.

    Play many random games and check if positions that "look similar"
    get different value labels. High noise means the value head is
    trying to fit contradictory targets.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Value Target Noise from Random Self-Play")
    print("=" * 70)

    # Collect (position_hash, target) pairs
    # position_hash = crude hash of the state_input
    position_data = {}  # hash -> list of targets
    total_positions = 0

    for g in range(n_games):
        state = game.new_game()
        trajectory = []
        while not state.terminal:
            inp = game.state_to_input(state)
            player = state.player
            mask = state.available_actions
            actions = np.where(mask == 1)[0]
            action = actions[np.random.randint(len(actions))]
            trajectory.append((inp, player))
            state = game.step(state, action)

        tv = state.terminal_value
        for inp, player in trajectory:
            target = tv * player
            # Crude hash: quantize input and hash
            h = hash(inp.tobytes())
            if h not in position_data:
                position_data[h] = []
            position_data[h].append(target)
            total_positions += 1

    # Analyze
    print(f"\n{total_positions} total positions from {n_games} random games")
    print(f"{len(position_data)} unique position hashes")

    # Check for contradictory targets (same position, different labels)
    contradictions = 0
    multi_hit = 0
    for h, targets in position_data.items():
        if len(targets) > 1:
            multi_hit += 1
            if len(set(targets)) > 1:
                contradictions += 1

    print(f"{multi_hit} positions seen multiple times")
    print(f"{contradictions} of those have contradictory targets (both +1 and -1)")

    # What fraction of total training examples come from contradicted positions?
    contradicted_examples = sum(len(v) for h, v in position_data.items() if len(set(v)) > 1)
    print(f"{contradicted_examples}/{total_positions} examples "
          f"({contradicted_examples/total_positions:.1%}) come from contradicted positions")

    # Target distribution analysis by game phase
    early_targets = []  # first 5 moves
    mid_targets = []    # moves 5-15
    late_targets = []   # moves 15+

    for g in range(n_games):
        state = game.new_game()
        trajectory = []
        while not state.terminal:
            trajectory.append(state.player)
            mask = state.available_actions
            actions = np.where(mask == 1)[0]
            action = actions[np.random.randint(len(actions))]
            state = game.step(state, action)

        tv = state.terminal_value
        for move_idx, player in enumerate(trajectory):
            target = tv * player
            if move_idx < 5:
                early_targets.append(target)
            elif move_idx < 15:
                mid_targets.append(target)
            else:
                late_targets.append(target)

    print(f"\nTarget distribution by game phase:")
    for name, targets in [("Early (0-4)", early_targets),
                          ("Mid (5-14)", mid_targets),
                          ("Late (15+)", late_targets)]:
        if targets:
            t = np.array(targets)
            print(f"  {name}: mean={t.mean():+.3f} std={t.std():.3f} "
                  f"+1: {(t>0).mean():.1%} -1: {(t<0).mean():.1%}")


def test_overfit_vs_generalize(game, net, n_train_games=30, n_test_games=30):
    """Test if the network has overfit to training positions.

    Generate two independent sets of positions and check prediction quality.
    If overfit, training-like positions may get better predictions than novel ones.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Generalization Check")
    print("=" * 70)

    def generate_positions(n_games):
        all_positions = []
        for _ in range(n_games):
            trajectory = []
            state = game.new_game()
            while not state.terminal:
                trajectory.append((game.state_to_input(state), state.player))
                mask = state.available_actions
                actions = np.where(mask == 1)[0]
                action = actions[np.random.randint(len(actions))]
                state = game.step(state, action)
            tv = state.terminal_value
            for inp, player in trajectory:
                all_positions.append((inp, tv * player))
        return all_positions

    set_a = generate_positions(n_train_games)
    set_b = generate_positions(n_test_games)

    def eval_set(positions, label):
        if not positions:
            return
        preds = []
        targets = []
        for inp, target in positions:
            v, _ = net.predict(inp)
            preds.append(v)
            targets.append(target)
        preds = np.array(preds)
        targets = np.array(targets)

        sign_acc = (np.sign(preds) == np.sign(targets)).mean()
        mae = np.abs(preds - targets).mean()
        corr = np.corrcoef(preds, targets)[0, 1] if len(preds) > 1 else 0
        sat = (np.abs(preds) > 0.95).mean()

        print(f"  {label}: sign_acc={sign_acc:.1%} MAE={mae:.3f} "
              f"corr={corr:+.3f} saturated={sat:.1%} "
              f"pred_mean={preds.mean():+.3f} pred_std={preds.std():.3f}")

    print(f"\nEvaluating on two independent random game sets...")
    eval_set(set_a, "Set A")
    eval_set(set_b, "Set B")
    print("  (If both are ~50% sign_acc, the network hasn't learned value at all)")
    print("  (If one is much better, the network has overfit to specific positions)")


def test_value_head_capacity(game, net):
    """Check value head internals: dead neurons, activation ranges."""
    print("\n" + "=" * 70)
    print("TEST 4: Value Head Internals")
    print("=" * 70)

    # Generate random positions
    positions = []
    for _ in range(50):
        state = game.new_game()
        while not state.terminal:
            positions.append(game.state_to_input(state))
            mask = state.available_actions
            actions = np.where(mask == 1)[0]
            action = actions[np.random.randint(len(actions))]
            state = game.step(state, action)

    inputs_t = torch.FloatTensor(np.array(positions[:200]))
    device = next(net.parameters()).device
    inputs_t = inputs_t.to(device)

    net.eval()
    with torch.no_grad():
        # Run through backbone
        import torch.nn.functional as F
        x = F.relu(net.bn(net.conv(inputs_t)))
        for block in net.res_blocks:
            x = block(x)

        # Value head layers
        v_conv = F.leaky_relu(net.value_bn(net.value_conv(x)), negative_slope=0.01)
        v_flat = v_conv.view(v_conv.size(0), -1)
        v_fc1 = F.leaky_relu(net.value_fc1(v_flat), negative_slope=0.01)
        v_pre_tanh = net.value_fc2(v_fc1)
        v_out = torch.tanh(v_pre_tanh)

    # Dead neurons in fc1
    fc1_np = v_fc1.cpu().numpy()
    dead_mask = (fc1_np == 0).all(axis=0)
    n_dead = dead_mask.sum()
    n_total = fc1_np.shape[1]

    # Activation stats
    alive_activations = fc1_np[:, ~dead_mask]

    # Pre-tanh range
    pre_tanh = v_pre_tanh.cpu().numpy().flatten()

    # Output distribution
    out = v_out.cpu().numpy().flatten()

    print(f"\nValue conv output: shape={list(v_conv.shape)} "
          f"mean={v_conv.mean():.4f} std={v_conv.std():.4f}")
    print(f"FC1: {n_dead}/{n_total} neurons dead ({n_dead/n_total:.1%})")
    if alive_activations.size > 0:
        print(f"  Alive neuron activations: mean={alive_activations.mean():.4f} "
              f"std={alive_activations.std():.4f} max={alive_activations.max():.4f}")
    print(f"Pre-tanh: range=[{pre_tanh.min():.2f}, {pre_tanh.max():.2f}] "
          f"mean={pre_tanh.mean():.4f} std={pre_tanh.std():.4f}")
    print(f"Output: mean={out.mean():.4f} std={out.std():.4f} "
          f"|v|={np.abs(out).mean():.4f} saturated={((np.abs(out) > 0.95).mean()):.1%}")

    # Effective rank of fc1 activations
    if alive_activations.shape[1] > 1:
        # SVD of alive activations
        U, S, Vt = np.linalg.svd(alive_activations - alive_activations.mean(axis=0), full_matrices=False)
        # Effective rank = exp(entropy of normalized singular values)
        s_norm = S / S.sum()
        s_norm = s_norm[s_norm > 1e-10]
        eff_rank = np.exp(-np.sum(s_norm * np.log(s_norm)))
        print(f"  Effective rank of fc1: {eff_rank:.1f}/{n_total - n_dead} alive neurons")
        print(f"  Top-5 singular values: {S[:5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="santorini")
    args = parser.parse_args()

    game = load_game(args.game)
    game_cfg = GAME_CONFIGS.get(args.game, {})

    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1))
    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=game_cfg.get("num_res_blocks", 3),
        num_filters=game_cfg.get("num_filters", 128),
        value_head_channels=game_cfg.get("value_head_channels", 2),
        value_head_fc_size=game_cfg.get("value_head_fc_size", 64),
        policy_head_channels=game_cfg.get("policy_head_channels", 2),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    checkpoint_dir = f"checkpoints/{args.game}"
    loaded = net.load_latest(checkpoint_dir)
    if loaded:
        print(f"Loaded: {loaded}")
    else:
        print("No checkpoint found, using untrained network.")

    # Run all tests
    test_symmetry_consistency(game, net)
    test_target_noise(game)
    test_overfit_vs_generalize(game, net)
    test_value_head_capacity(game, net)


if __name__ == "__main__":
    main()
