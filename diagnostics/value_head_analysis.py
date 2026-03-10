"""Diagnostic E: Value head activation analysis.

Analyzes the value head internals:
- Hidden layer activations (dead neurons, saturation, effective rank)
- Weight matrix statistics
- Per-player activation patterns

Run: python -m diagnostics.value_head_analysis [--game connect4]
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from network import AlphaZeroNet
from game_configs import GAME_CONFIGS
from train import load_game
from games.connect4 import GameState as C4State


def generate_positions(game, num_games=100):
    """Generate positions from random games with known outcomes."""
    examples = []
    rng = np.random.RandomState(42)

    for _ in range(num_games):
        s = game.new_game()
        trajectory = []
        while not s.terminal:
            trajectory.append((game.state_to_input(s), s.player))
            avail = np.nonzero(s.available_actions)[0]
            s = game.step(s, rng.choice(avail))
        tv = s.terminal_value
        for inp, player in trajectory:
            examples.append((inp, tv * player, player))  # input, target, player
    return examples


def analyze_value_head(net, examples, device):
    """Run positions through value head and analyze activations."""
    net.eval()
    inputs = torch.FloatTensor(np.array([e[0] for e in examples])).to(device)
    targets = np.array([e[1] for e in examples])
    players = np.array([e[2] for e in examples])

    # Hook to capture intermediate activations
    activations = {}

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook

    hooks = []
    hooks.append(net.value_conv.register_forward_hook(hook_fn("value_conv")))
    hooks.append(net.value_bn.register_forward_hook(hook_fn("value_bn")))
    hooks.append(net.value_fc1.register_forward_hook(hook_fn("value_fc1")))
    hooks.append(net.value_fc2.register_forward_hook(hook_fn("value_fc2")))

    with torch.no_grad():
        pred_vs, _ = net(inputs)

    for h in hooks:
        h.remove()

    preds = pred_vs.squeeze(1).cpu().numpy()

    # === 1. Weight statistics ===
    print("=" * 70)
    print("VALUE HEAD WEIGHT STATISTICS")
    print("=" * 70)

    for name in ["value_conv", "value_bn", "value_fc1", "value_fc2"]:
        module = getattr(net, name)
        for pname, param in module.named_parameters():
            p = param.detach().cpu().numpy()
            print(f"  {name}.{pname:<12} shape={str(list(p.shape)):<15} "
                  f"mean={p.mean():+.6f} std={p.std():.6f} "
                  f"min={p.min():+.6f} max={p.max():+.6f}")

    # === 2. Activation statistics ===
    print(f"\n{'='*70}")
    print("VALUE HEAD ACTIVATION STATISTICS")
    print("=" * 70)

    # After value_conv + bn + relu (before flattening)
    act_conv = F.relu(activations["value_bn"])
    act_conv_flat = act_conv.view(act_conv.size(0), -1)
    print(f"\n  After value_conv+bn+relu: shape={list(act_conv.shape)}")
    print(f"    mean={act_conv_flat.mean():.4f} std={act_conv_flat.std():.4f}")
    print(f"    zeros (dead)={float((act_conv_flat == 0).float().mean()):.1%}")
    print(f"    range=[{act_conv_flat.min():.4f}, {act_conv_flat.max():.4f}]")

    # After fc1 + relu
    act_fc1_pre = activations["value_fc1"]
    act_fc1 = F.relu(act_fc1_pre)
    print(f"\n  After value_fc1 (pre-relu): shape={list(act_fc1_pre.shape)}")
    print(f"    mean={act_fc1_pre.mean():.4f} std={act_fc1_pre.std():.4f}")
    print(f"    range=[{act_fc1_pre.min():.4f}, {act_fc1_pre.max():.4f}]")
    print(f"  After value_fc1 (post-relu):")
    print(f"    mean={act_fc1.mean():.4f} std={act_fc1.std():.4f}")
    dead_neurons = (act_fc1 == 0).float().mean(dim=0)  # per-neuron dead rate
    print(f"    dead neurons (>99% zero across batch): {int((dead_neurons > 0.99).sum())}/32")
    print(f"    mostly dead (>90% zero): {int((dead_neurons > 0.90).sum())}/32")
    print(f"    mostly active (<10% zero): {int((dead_neurons < 0.10).sum())}/32")
    print(f"    per-neuron dead rates: {dead_neurons.numpy().round(2).tolist()}")

    # After fc2 + tanh
    act_fc2 = activations["value_fc2"]
    act_tanh = torch.tanh(act_fc2)
    print(f"\n  After value_fc2 (pre-tanh): shape={list(act_fc2.shape)}")
    print(f"    mean={act_fc2.mean():.4f} std={act_fc2.std():.4f}")
    print(f"    range=[{act_fc2.min():.4f}, {act_fc2.max():.4f}]")
    print(f"  After tanh:")
    print(f"    mean={act_tanh.mean():.4f} std={act_tanh.std():.4f}")
    print(f"    saturated (|v|>0.95): {float((act_tanh.abs() > 0.95).float().mean()):.1%}")

    # === 3. Effective rank of fc1 weights ===
    print(f"\n{'='*70}")
    print("EFFECTIVE RANK ANALYSIS")
    print("=" * 70)

    fc1_w = net.value_fc1.weight.detach().cpu().numpy()  # (32, 42)
    U, S, Vt = np.linalg.svd(fc1_w, full_matrices=False)
    S_normalized = S / S.sum()
    entropy = -np.sum(S_normalized * np.log(S_normalized + 1e-10))
    effective_rank = np.exp(entropy)
    print(f"  value_fc1 weight matrix ({fc1_w.shape[0]}x{fc1_w.shape[1]}):")
    print(f"    Singular values: {S.round(4).tolist()}")
    print(f"    Top-5 explain: {S[:5].sum()/S.sum():.1%} of total variance")
    print(f"    Effective rank: {effective_rank:.1f} / {min(fc1_w.shape)}")
    print(f"    Condition number: {S[0]/max(S[-1], 1e-10):.1f}")

    # === 4. Per-player activation patterns ===
    print(f"\n{'='*70}")
    print("PER-PLAYER ACTIVATION PATTERNS")
    print("=" * 70)

    x_mask = players == -1
    o_mask = players == 1

    fc1_acts = act_fc1.numpy()
    for p_mask, pname in [(x_mask, "X-to-move"), (o_mask, "O-to-move")]:
        p_acts = fc1_acts[p_mask]
        p_preds = preds[p_mask]
        p_targets = targets[p_mask]
        print(f"\n  {pname} ({p_mask.sum()} positions):")
        print(f"    fc1 activation mean: {p_acts.mean():.4f}")
        print(f"    fc1 activation std:  {p_acts.std():.4f}")
        print(f"    pred value mean:     {p_preds.mean():+.4f}")
        print(f"    target value mean:   {p_targets.mean():+.4f}")
        # Per-neuron mean difference between X and O
        if pname == "X-to-move":
            x_neuron_means = p_acts.mean(axis=0)
        else:
            o_neuron_means = p_acts.mean(axis=0)

    if x_mask.any() and o_mask.any():
        diff = x_neuron_means - o_neuron_means
        print(f"\n  Neuron activation difference (X-mean minus O-mean):")
        print(f"    max_diff={diff.max():.4f} min_diff={diff.min():.4f} "
              f"mean_abs_diff={np.abs(diff).mean():.4f}")
        top_diff_idx = np.argsort(np.abs(diff))[::-1][:5]
        print(f"    Top discriminating neurons: "
              f"{[(int(i), f'{diff[i]:+.4f}') for i in top_diff_idx]}")

    # === 5. Correlation between fc1 activations and correct value ===
    print(f"\n{'='*70}")
    print("FC1 NEURON-VALUE CORRELATIONS")
    print("=" * 70)

    correlations = []
    for neuron_idx in range(fc1_acts.shape[1]):
        if fc1_acts[:, neuron_idx].std() > 1e-6:
            corr = np.corrcoef(fc1_acts[:, neuron_idx], targets)[0, 1]
        else:
            corr = 0.0
        correlations.append(corr)
    correlations = np.array(correlations)

    print(f"  Correlations of each fc1 neuron with target value:")
    print(f"    {correlations.round(3).tolist()}")
    print(f"    max |corr|={np.abs(correlations).max():.3f}")
    print(f"    mean |corr|={np.abs(correlations).mean():.3f}")
    useful = (np.abs(correlations) > 0.1).sum()
    print(f"    neurons with |corr|>0.1: {useful}/32")

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    dead_count = int((dead_neurons > 0.99).sum())
    print(f"  Dead neurons: {dead_count}/32")
    print(f"  Effective rank: {effective_rank:.1f}/{min(fc1_w.shape)}")
    print(f"  Useful neurons (|corr|>0.1): {useful}/32")
    print(f"  Pred mean X={preds[x_mask].mean():+.3f} O={preds[o_mask].mean():+.3f}")
    print(f"  Target mean X={targets[x_mask].mean():+.3f} O={targets[o_mask].mean():+.3f}")

    if dead_count > 10:
        print("\n  WARNING: Many dead neurons — capacity severely reduced!")
    if effective_rank < 10:
        print(f"\n  WARNING: Low effective rank ({effective_rank:.1f}) — "
              f"weight matrix not using full capacity!")
    if useful < 5:
        print(f"\n  WARNING: Few neurons correlate with value — "
              f"value head may be unable to extract value signal!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="connect4")
    parser.add_argument("--num-games", type=int, default=100)
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
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    loaded = net.load_latest(f"checkpoints/{args.game}")
    if loaded:
        print(f"Loaded: {loaded}")
    else:
        print("No checkpoint, using untrained network")

    examples = generate_positions(game, args.num_games)
    print(f"Generated {len(examples)} positions from {args.num_games} games\n")

    analyze_value_head(net, examples, device)
