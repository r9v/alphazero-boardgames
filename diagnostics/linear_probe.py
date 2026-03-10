"""Diagnostic D: Linear probe test.

Extracts features from the residual backbone and trains a simple linear
regression to predict values. This tests whether the backbone features
contain enough information for correct value prediction.

- If linear probe succeeds: features are good, value head architecture is bad
- If linear probe fails: backbone features themselves are insufficient

Run: python -m diagnostics.linear_probe [--game connect4]
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import AlphaZeroNet
from game_configs import GAME_CONFIGS
from train import load_game
from games.connect4 import GameState as C4State


def extract_backbone_features(net, states_tensor, device):
    """Extract features from after the residual blocks (before value/policy heads)."""
    net.eval()
    with torch.no_grad():
        x = states_tensor.to(device)
        # Run through backbone
        x = F.relu(net.bn(net.conv(x)))
        for block in net.res_blocks:
            x = block(x)
        # x is now (batch, num_filters, H, W)
        # Also extract value head intermediate
        v = F.relu(net.value_bn(net.value_conv(x)))
        v_flat = v.view(v.size(0), -1)  # (batch, H*W)
    return x.cpu(), v_flat.cpu()


def generate_training_data(game, num_games=200):
    """Play random games and collect (state_input, value) pairs."""
    examples = []
    rng = np.random.RandomState(123)

    for _ in range(num_games):
        s = game.new_game()
        trajectory = []
        while not s.terminal:
            trajectory.append((game.state_to_input(s), s.player))
            avail = np.nonzero(s.available_actions)[0]
            s = game.step(s, rng.choice(avail))

        tv = s.terminal_value
        for inp, player in trajectory:
            target = tv * player  # relative value
            examples.append((inp, target))

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="connect4")
    parser.add_argument("--num-games", type=int, default=200)
    args = parser.parse_args()

    game = load_game(args.game)
    game_cfg = GAME_CONFIGS.get(args.game, {})
    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1))

    # Load trained network if available
    net = AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=game_cfg.get("num_res_blocks", 3),
        num_filters=game_cfg.get("num_filters", 128),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    checkpoint_dir = f"checkpoints/{args.game}"
    loaded = net.load_latest(checkpoint_dir)
    if loaded:
        print(f"Loaded checkpoint: {loaded}")
    else:
        print("No checkpoint found, using UNTRAINED network")

    # Generate data
    print(f"\nGenerating {args.num_games} random games for probe training...")
    examples = generate_training_data(game, args.num_games)
    print(f"  Got {len(examples)} position-value pairs")

    # Split into train/test
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(examples))
    split = int(0.8 * len(examples))
    train_idx, test_idx = indices[:split], indices[split:]

    all_inputs = np.array([e[0] for e in examples])
    all_targets = np.array([e[1] for e in examples])
    states_tensor = torch.FloatTensor(all_inputs)

    # Extract features
    print("Extracting backbone features...")
    backbone_feats, vhead_feats = extract_backbone_features(net, states_tensor, device)

    # Flatten backbone features
    backbone_flat = backbone_feats.view(backbone_feats.size(0), -1)  # (N, filters*H*W)
    print(f"  Backbone features shape: {list(backbone_flat.shape)}")
    print(f"  Value head input shape: {list(vhead_feats.shape)}")

    targets_tensor = torch.FloatTensor(all_targets)

    # === Test 1: Linear probe on value head input (H*W features after 1x1 conv) ===
    print(f"\n{'='*60}")
    print("TEST 1: Linear probe on value head input (after 1x1 conv)")
    print("=" * 60)
    _train_probe(vhead_feats, targets_tensor, train_idx, test_idx, "vhead_input")

    # === Test 2: Linear probe on full backbone features ===
    print(f"\n{'='*60}")
    print("TEST 2: Linear probe on full backbone (all filters * H * W)")
    print("=" * 60)
    _train_probe(backbone_flat, targets_tensor, train_idx, test_idx, "backbone")

    # === Test 3: Compare with actual value head ===
    print(f"\n{'='*60}")
    print("TEST 3: Actual value head predictions")
    print("=" * 60)
    net.eval()
    with torch.no_grad():
        preds, _ = net(states_tensor.to(device))
        preds = preds.squeeze(1).cpu().numpy()

    test_preds = preds[test_idx]
    test_targets = all_targets[test_idx]
    mse = np.mean((test_preds - test_targets) ** 2)
    sign_acc = np.mean(np.sign(test_preds) == np.sign(test_targets))
    corr = np.corrcoef(test_preds, test_targets)[0, 1] if len(test_preds) > 1 else 0

    # Per-player breakdown
    all_players = []
    for e in examples:
        # Infer player from piece counts
        inp = e[0]
        num_hist = getattr(game, 'num_history_states', 2)
        ch = 2 * num_hist
        my_count = inp[ch].sum()
        opp_count = inp[ch + 1].sum()
        all_players.append(-1 if my_count == opp_count else 1)
    all_players = np.array(all_players)

    test_players = all_players[test_idx]
    for p, pname in [(-1, "X"), (1, "O")]:
        mask = test_players == p
        if mask.any():
            p_preds = test_preds[mask]
            p_tgts = test_targets[mask]
            p_mse = np.mean((p_preds - p_tgts) ** 2)
            p_sign = np.mean(np.sign(p_preds) == np.sign(p_tgts))
            p_corr = np.corrcoef(p_preds, p_tgts)[0, 1] if len(p_preds) > 1 else 0
            print(f"  {pname}: MSE={p_mse:.4f} sign_acc={p_sign:.1%} corr={p_corr:+.3f} "
                  f"pred_mean={p_preds.mean():+.3f} target_mean={p_tgts.mean():+.3f}")

    print(f"\n  Overall: MSE={mse:.4f} sign_acc={sign_acc:.1%} corr={corr:+.3f}")


def _train_probe(features, targets, train_idx, test_idx, name, steps=1000, lr=0.01):
    """Train a linear probe and report accuracy."""
    feat_dim = features.shape[1]
    probe = nn.Linear(feat_dim, 1)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    train_x = features[train_idx]
    train_y = targets[train_idx].unsqueeze(1)
    test_x = features[test_idx]
    test_y = targets[test_idx]

    for step in range(steps):
        pred = torch.tanh(probe(train_x))
        loss = F.mse_loss(pred, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                test_pred = torch.tanh(probe(test_x)).squeeze(1).numpy()
                test_tgt = test_y.numpy()
                test_mse = np.mean((test_pred - test_tgt) ** 2)
                sign_acc = np.mean(np.sign(test_pred) == np.sign(test_tgt))
                corr = np.corrcoef(test_pred, test_tgt)[0, 1] if len(test_pred) > 1 else 0
            print(f"  Step {step:>4}: train_loss={loss.item():.4f} "
                  f"test_MSE={test_mse:.4f} sign_acc={sign_acc:.1%} corr={corr:+.3f}")

    # Final verdict
    with torch.no_grad():
        test_pred = torch.tanh(probe(test_x)).squeeze(1).numpy()
        test_tgt = test_y.numpy()
        final_corr = np.corrcoef(test_pred, test_tgt)[0, 1] if len(test_pred) > 1 else 0

    if final_corr > 0.3:
        print(f"  VERDICT [{name}]: Linear probe SUCCEEDS (corr={final_corr:+.3f}). "
              f"Features contain value information.")
    elif final_corr > 0.0:
        print(f"  VERDICT [{name}]: Weak signal (corr={final_corr:+.3f}). "
              f"Features have some value info but not strong.")
    else:
        print(f"  VERDICT [{name}]: Linear probe FAILS (corr={final_corr:+.3f}). "
              f"Features lack value information.")


if __name__ == "__main__":
    main()
