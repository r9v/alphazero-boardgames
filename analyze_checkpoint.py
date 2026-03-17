"""
Analyze checkpoint to diagnose backbone/value head health.

Usage:
    python analyze_checkpoint.py [checkpoint_path]
    python analyze_checkpoint.py                          # uses latest checkpoint
    python analyze_checkpoint.py checkpoints/santorini/20260311-224050.pt
"""
import torch
import numpy as np
import sys
import os

def find_latest_checkpoint(game="santorini"):
    """Find the latest checkpoint for a game."""
    ckpt_dir = f"checkpoints/{game}"
    latest_file = os.path.join(ckpt_dir, "latest.txt")
    if os.path.exists(latest_file):
        with open(latest_file) as f:
            return os.path.join(ckpt_dir, f.read().strip())
    # Fallback: find most recent .pt file
    pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if pts:
        return os.path.join(ckpt_dir, pts[-1])
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

def load_checkpoint(path):
    """Load checkpoint and extract state dict."""
    print(f"Loading {path}...")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state

def detect_architecture(sd):
    """Auto-detect architecture params from state dict."""
    num_filters = sd["conv.weight"].shape[0]
    input_channels = sd["conv.weight"].shape[1]
    num_res_blocks = sum(1 for k in sd if k.startswith("res_blocks.") and k.endswith(".conv1.weight"))
    value_head_channels = sd["value_conv.weight"].shape[0]
    fc1_in = sd["value_fc1.weight"].shape[1]
    fc1_out = sd["value_fc1.weight"].shape[0]
    board_area = fc1_in // value_head_channels
    action_size = sd["policy_fc.weight"].shape[0]
    return {
        "num_filters": num_filters,
        "input_channels": input_channels,
        "num_res_blocks": num_res_blocks,
        "value_head_channels": value_head_channels,
        "value_head_fc_size": fc1_out,
        "board_area": board_area,
        "action_size": action_size,
    }

def section_backbone_health(sd, arch):
    """Section 1: Per-channel weight magnitude analysis."""
    N = arch["num_filters"]
    R = arch["num_res_blocks"]

    print("\n" + "="*80)
    print("SECTION 1: BACKBONE CHANNEL HEALTH")
    print("="*80)

    init_conv_norms = sd["conv.weight"].flatten(1).norm(dim=1)

    rb_norms = []
    for i in range(R):
        c1 = sd[f"res_blocks.{i}.conv1.weight"]
        c2 = sd[f"res_blocks.{i}.conv2.weight"]
        c1_out = c1.flatten(1).norm(dim=1)
        c2_out = c2.flatten(1).norm(dim=1)
        c1_in = c1.permute(1,0,2,3).flatten(1).norm(dim=1)
        c2_in = c2.permute(1,0,2,3).flatten(1).norm(dim=1)
        rb_norms.append((c1_out, c2_out, c1_in, c2_in))

    composite = init_conv_norms.clone()
    for c1_out, c2_out, c1_in, c2_in in rb_norms:
        composite += c1_out + c2_out + c1_in + c2_in

    sorted_ch = composite.argsort()
    print(f"\nComposite weight norm per backbone channel (sum across all layers):")
    print(f"  Min: ch{sorted_ch[0].item()} = {composite[sorted_ch[0]].item():.4f}")
    print(f"  p10: {composite.quantile(0.1).item():.4f}")
    print(f"  p50: {composite.quantile(0.5).item():.4f}")
    print(f"  p90: {composite.quantile(0.9).item():.4f}")
    print(f"  Max: ch{sorted_ch[-1].item()} = {composite[sorted_ch[-1]].item():.4f}")
    print(f"  Ratio max/min: {composite.max().item() / composite.min().item():.1f}x")

    # Per-layer breakdown for bottom/top channels
    header = f"  {'ch':>4} | {'init_conv':>9}"
    for i in range(R):
        header += f" | {'rb'+str(i)+'_c1':>7} | {'rb'+str(i)+'_c2':>7}"
    header += f" | {'composite':>9}"

    for label, indices in [("bottom 10", sorted_ch[:10].tolist()), ("top 10", sorted_ch[-10:].flip(0).tolist())]:
        print(f"\nPer-layer channel norms ({label} by composite):")
        print(header)
        for ch in indices:
            row = f"  {ch:>4} | {init_conv_norms[ch].item():>9.4f}"
            for c1_out, c2_out, _, _ in rb_norms:
                row += f" | {c1_out[ch].item():>7.4f} | {c2_out[ch].item():>7.4f}"
            row += f" | {composite[ch].item():>9.4f}"
            print(row)

    return composite, init_conv_norms, rb_norms

def section_policy_vs_value(sd, composite):
    """Section 2: Which backbone channels does each head use?"""
    print("\n" + "="*80)
    print("SECTION 2: POLICY vs VALUE BACKBONE USAGE")
    print("="*80)

    # Reshape to [out_channels, backbone_channels] handling 1-channel case
    vc_w = sd["value_conv.weight"].squeeze(-1).squeeze(-1)  # [V, N]
    pc_w = sd["policy_conv.weight"].squeeze(-1).squeeze(-1)  # [P, N]
    if vc_w.dim() == 1:
        vc_w = vc_w.unsqueeze(0)  # [1, N] for single value channel
    if pc_w.dim() == 1:
        pc_w = pc_w.unsqueeze(0)
    value_per_bb = vc_w.abs().sum(dim=0)   # [N]
    policy_per_bb = pc_w.abs().sum(dim=0)  # [N]

    print(f"\nValue conv total weight per backbone channel:")
    print(f"  Sum: {value_per_bb.sum().item():.4f}  Mean: {value_per_bb.mean().item():.4f}  Max: {value_per_bb.max().item():.4f}")
    print(f"Policy conv total weight per backbone channel:")
    print(f"  Sum: {policy_per_bb.sum().item():.4f}  Mean: {policy_per_bb.mean().item():.4f}  Max: {policy_per_bb.max().item():.4f}")

    N = len(composite)
    topK = min(10, N)
    v_topK = value_per_bb.argsort(descending=True)[:topK].tolist()
    p_topK = policy_per_bb.argsort(descending=True)[:topK].tolist()
    overlap = set(v_topK) & set(p_topK)
    print(f"\nTop-{topK} channels for value:  {v_topK}")
    print(f"Top-{topK} channels for policy: {p_topK}")
    print(f"Overlap in top-{topK}: {len(overlap)} channels: {overlap}")

    ranks = composite.argsort(descending=True).tolist()
    print(f"\nValue's top-{topK} channels health:")
    for ch in v_topK:
        r = ranks.index(ch) + 1
        print(f"  ch{ch:>3}: val_w={value_per_bb[ch].item():.4f}  pol_w={policy_per_bb[ch].item():.4f}  bb_norm={composite[ch].item():.2f}  rank={r}/{N}")

    print(f"\nPolicy's top-{topK} channels health:")
    for ch in p_topK:
        r = ranks.index(ch) + 1
        print(f"  ch{ch:>3}: val_w={value_per_bb[ch].item():.4f}  pol_w={policy_per_bb[ch].item():.4f}  bb_norm={composite[ch].item():.2f}  rank={r}/{N}")

    corr_vh = np.corrcoef(value_per_bb.numpy(), composite.numpy())[0,1]
    corr_ph = np.corrcoef(policy_per_bb.numpy(), composite.numpy())[0,1]
    corr_vp = np.corrcoef(value_per_bb.numpy(), policy_per_bb.numpy())[0,1]
    print(f"\nCorrelation(value_attn, bb_health):  {corr_vh:+.4f}")
    print(f"Correlation(policy_attn, bb_health): {corr_ph:+.4f}")
    print(f"Correlation(value_attn, policy_attn): {corr_vp:+.4f}")

    # Quartile analysis
    N = len(composite)
    q_size = N // 4
    sorted_by_health = composite.argsort()
    q1 = sorted_by_health[:q_size]
    q4 = sorted_by_health[-q_size:]
    print(f"\nValue attention by backbone health quartile:")
    print(f"  Q1 (weakest {q_size}):  mean val_w={value_per_bb[q1].mean().item():.4f}  mean pol_w={policy_per_bb[q1].mean().item():.4f}")
    print(f"  Q4 (strongest {q_size}): mean val_w={value_per_bb[q4].mean().item():.4f}  mean pol_w={policy_per_bb[q4].mean().item():.4f}")
    v_ratio = value_per_bb[q4].mean().item() / max(value_per_bb[q1].mean().item(), 1e-8)
    p_ratio = policy_per_bb[q4].mean().item() / max(policy_per_bb[q1].mean().item(), 1e-8)
    print(f"  Ratio Q4/Q1 value: {v_ratio:.2f}x")
    print(f"  Ratio Q4/Q1 policy: {p_ratio:.2f}x")

    return value_per_bb, policy_per_bb

def section_fc1_analysis(sd):
    """Section 3: FC1 neuron importance and death analysis."""
    print("\n" + "="*80)
    print("SECTION 3: FC1 WEIGHT ANALYSIS")
    print("="*80)

    fc1_w = sd["value_fc1.weight"]
    fc1_b = sd["value_fc1.bias"]
    fc2_w = sd["value_fc2.weight"]

    print(f"\nFC1 weight shape: {fc1_w.shape}")
    print(f"FC2 weight shape: {fc2_w.shape}")

    fc1_in_norms = fc1_w.norm(dim=1)
    fc2_out_w = fc2_w.squeeze().abs()
    importance = fc1_in_norms * fc2_out_w
    imp_sorted = importance.argsort(descending=True)

    print(f"\nFC1 neuron importance (in_norm * |fc2_w|):")
    print(f"  Top 10:")
    for i in range(10):
        n = imp_sorted[i].item()
        print(f"    neuron {n:>3}: importance={importance[n].item():.4f}  in_norm={fc1_in_norms[n].item():.4f}  fc2_w={fc2_w[0,n].item():+.4f}  bias={fc1_b[n].item():+.4f}")
    print(f"  Bottom 10:")
    for i in range(-10, 0):
        n = imp_sorted[i].item()
        print(f"    neuron {n:>3}: importance={importance[n].item():.2e}  in_norm={fc1_in_norms[n].item():.2e}  fc2_w={fc2_w[0,n].item():+.2e}")

    print(f"\n  Importance distribution:")
    print(f"    p10={importance.quantile(0.1).item():.6f}  p50={importance.quantile(0.5).item():.4f}  p90={importance.quantile(0.9).item():.4f}  max={importance.max().item():.4f}")

    imp_sorted_vals = importance[imp_sorted]
    cumsum = imp_sorted_vals.cumsum(0) / imp_sorted_vals.sum()
    for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
        n_needed = (cumsum <= pct).sum().item() + 1
        print(f"    {pct*100:.0f}% of total importance carried by top {n_needed} neurons")

    # Dead neuron analysis
    truly_zero = (fc1_in_norms < 1e-6).sum().item()
    very_small = (fc1_in_norms < 0.01).sum().item()
    alive = (fc1_in_norms > 0.1).sum().item()
    print(f"\n  Dead neuron analysis:")
    print(f"    Zero weights (<1e-6 norm): {truly_zero}/{len(fc1_in_norms)}")
    print(f"    Very small weights (<0.01 norm): {very_small}/{len(fc1_in_norms)}")
    print(f"    Alive neurons (>0.1 norm): {alive}/{len(fc1_in_norms)}")

    alive_mask = fc1_in_norms > 0.1
    dead_mask = fc1_in_norms < 0.01
    if alive_mask.sum() > 0:
        print(f"    Alive neuron biases: mean={fc1_b[alive_mask].mean().item():.4f}  min={fc1_b[alive_mask].min().item():.4f}")
    if dead_mask.sum() > 0:
        print(f"    Dead neuron biases:  mean={fc1_b[dead_mask].mean().item():.4f}  min={fc1_b[dead_mask].min().item():.4f}")

    return importance, imp_sorted

def section_value_conv(sd):
    """Section 4: Value conv channel analysis with BN."""
    print("\n" + "="*80)
    print("SECTION 4: VALUE CONV CHANNEL ANALYSIS")
    print("="*80)

    vc_w = sd["value_conv.weight"].squeeze(-1).squeeze(-1)  # [V, N]
    if vc_w.dim() == 1:
        vc_w = vc_w.unsqueeze(0)  # [1, N] for single value channel
    v_bn_gamma = sd["value_bn.weight"]
    v_bn_beta = sd["value_bn.bias"]
    v_bn_var = sd.get("value_bn.running_var")  # None for GroupNorm checkpoints
    v_bn_mean = sd.get("value_bn.running_mean")
    num_ch = vc_w.shape[0]

    norm_type = "BN" if v_bn_var is not None else "GN"
    print(f"\nValue {norm_type} parameters per channel:")
    if v_bn_var is not None:
        print(f"  {'ch':>3} | {'gamma':>7} | {'beta':>7} | {'var':>8} | {'mean':>8} | {'eff_gain':>8}")
        for i in range(num_ch):
            eff_gain = v_bn_gamma[i].item() / (v_bn_var[i].item() + 1e-5)**0.5
            print(f"  {i:>3} | {v_bn_gamma[i].item():>7.4f} | {v_bn_beta[i].item():>7.4f} | {v_bn_var[i].item():>8.6f} | {v_bn_mean[i].item():>8.6f} | {eff_gain:>8.2f}")
    else:
        print(f"  {'ch':>3} | {'gamma':>7} | {'beta':>7}")
        for i in range(num_ch):
            print(f"  {i:>3} | {v_bn_gamma[i].item():>7.4f} | {v_bn_beta[i].item():>7.4f}")

    print(f"\nValue conv attention spread (per output channel):")
    for i in range(num_ch):
        ch_w = vc_w[i].abs()
        ch_sorted = ch_w.argsort(descending=True)
        top5_pct = ch_w[ch_sorted[:5]].sum().item() / ch_w.sum().item() * 100
        top20_pct = ch_w[ch_sorted[:20]].sum().item() / ch_w.sum().item() * 100
        eff_str = ""
        if v_bn_var is not None:
            eff_gain = v_bn_gamma[i].item() / (v_bn_var[i].item() + 1e-5)**0.5
            eff_str = f"  eff_gain={eff_gain:.1f}"
        print(f"  ch{i}: top5={top5_pct:.1f}%  top20={top20_pct:.1f}%  total_w={ch_w.sum().item():.3f}{eff_str}  top3_bb={ch_sorted[:3].tolist()}")

def section_svd_rank(sd, arch):
    """Section 5: SVD effective rank analysis."""
    R = arch["num_res_blocks"]
    N = arch["num_filters"]

    print("\n" + "="*80)
    print("SECTION 5: SVD EFFECTIVE RANK (IS THE MODEL TOO LARGE?)")
    print("="*80)

    print(f"\nSingular value analysis of residual block conv weights:")
    for i in range(R):
        for j, cname in [(1, "conv1"), (2, "conv2")]:
            w = sd[f"res_blocks.{i}.{cname}.weight"].flatten(1)
            svs = torch.linalg.svdvals(w)
            energy = (svs**2).cumsum(0) / (svs**2).sum()
            rank_90 = (energy <= 0.9).sum().item() + 1
            rank_99 = (energy <= 0.99).sum().item() + 1
            near_zero = (svs < 1e-6).sum().item()
            sv_min = max(svs[-1].item(), 1e-12)
            mid = N // 2
            print(f"  rb{i}.{cname}: rank_90={rank_90}/{N}  rank_99={rank_99}/{N}  near_zero_sv={near_zero}  sv[0]={svs[0].item():.3f}  sv[{mid}]={svs[mid].item():.6f}  sv[-1]={svs[-1].item():.2e}")

def section_bn_analysis(sd, arch):
    """Section 6: Batch norm effective gain analysis across layers."""
    R = arch["num_res_blocks"]
    N = arch["num_filters"]

    print("\n" + "="*80)
    print("SECTION 6: BATCH NORM HEALTH ACROSS LAYERS")
    print("="*80)

    layers = ["bn"]
    for i in range(R):
        layers.extend([f"res_blocks.{i}.bn1", f"res_blocks.{i}.bn2"])

    has_running_var = f"bn.running_var" in sd
    if has_running_var:
        print(f"\nBN effective gain (gamma / sqrt(var + eps)) per layer:")
        for layer_name in layers:
            gamma = sd[f"{layer_name}.weight"]
            var = sd[f"{layer_name}.running_var"]
            eff_gain = gamma / (var + 1e-5).sqrt()
            near_zero = (eff_gain.abs() < 0.1).sum().item()
            neg = (gamma < -0.01).sum().item()
            print(f"  {layer_name:>22}: |gain| mean={eff_gain.abs().mean():.2f}  near_zero(<0.1)={near_zero}/{N}  neg_gamma={neg}  range=[{eff_gain.min():.2f}, {eff_gain.max():.2f}]")
    else:
        print(f"\nGroupNorm gamma per layer (no running stats):")
        for layer_name in layers:
            gamma = sd[f"{layer_name}.weight"]
            near_zero = (gamma.abs() < 0.1).sum().item()
            neg = (gamma < -0.01).sum().item()
            print(f"  {layer_name:>22}: |gamma| mean={gamma.abs().mean():.2f}  near_zero(<0.1)={near_zero}/{N}  neg_gamma={neg}  range=[{gamma.min():.2f}, {gamma.max():.2f}]")

    # Overlap of dead channels across res block outputs (bn2/gn2)
    dead_sets = []
    for i in range(R):
        layer_name = f"res_blocks.{i}.bn2"
        gamma = sd[f"{layer_name}.weight"]
        if has_running_var:
            var = sd[f"{layer_name}.running_var"]
            eff_gain = gamma / (var + 1e-5).sqrt()
            dead_idx = (eff_gain.abs() < 0.1).nonzero().squeeze(-1).tolist()
        else:
            dead_idx = (gamma.abs() < 0.1).nonzero().squeeze(-1).tolist()
        if isinstance(dead_idx, int):
            dead_idx = [dead_idx]
        dead_sets.append(set(dead_idx))

    print(f"\nNorm dead channel overlap ({'|gain|' if has_running_var else '|gamma|'} < 0.1) across res block outputs:")
    for i, s in enumerate(dead_sets):
        print(f"  rb{i}.bn2: {len(s)} dead")
    if R >= 2:
        for i in range(R - 1):
            overlap = dead_sets[i] & dead_sets[i+1]
            print(f"  rb{i}+rb{i+1} overlap: {len(overlap)}")
    if R >= 3:
        all_overlap = dead_sets[0] & dead_sets[1] & dead_sets[2]
        print(f"  all blocks overlap: {len(all_overlap)}")

def section_policy_head(sd, arch):
    """Section 7: Policy head internal analysis."""
    N = arch["num_filters"]
    board_area = arch["board_area"]
    action_size = arch["action_size"]

    print("\n" + "="*80)
    print("SECTION 7: POLICY HEAD ANALYSIS")
    print("="*80)

    # --- 7a: Policy conv channel utilization ---
    pc_w = sd["policy_conv.weight"]  # shape: [P, N, 1, 1]
    pc_w_sq = pc_w.squeeze(-1).squeeze(-1)  # shape: [P, N]
    if pc_w_sq.dim() == 1:
        pc_w_sq = pc_w_sq.unsqueeze(0)
    policy_channels = pc_w_sq.shape[0]
    print(f"\nPolicy conv weight shape: {list(pc_w.shape)}")
    print(f"  (maps {N} backbone channels -> {policy_channels} policy channels via 1x1 conv)")

    for ch in range(pc_w_sq.shape[0]):
        ch_w = pc_w_sq[ch]
        ch_abs = ch_w.abs()
        top5_idx = ch_abs.argsort(descending=True)[:5]
        active = (ch_abs > 0.01).sum().item()
        strong = (ch_abs > 0.1).sum().item()
        print(f"\n  Policy conv ch{ch}:")
        print(f"    L1 norm: {ch_abs.sum().item():.4f}  L2 norm: {ch_w.norm().item():.4f}")
        print(f"    Active (>0.01): {active}/{N}  Strong (>0.1): {strong}/{N}")
        print(f"    Top-5 backbone channels: {top5_idx.tolist()}")
        print(f"    Top-5 weights: [{', '.join(f'{ch_w[i].item():+.4f}' for i in top5_idx)}]")
        # Concentration: what fraction of total weight is in top-K channels
        sorted_abs = ch_abs.sort(descending=True).values
        cum = sorted_abs.cumsum(0) / sorted_abs.sum()
        for pct in [0.5, 0.8, 0.9]:
            n_needed = (cum <= pct).sum().item() + 1
            print(f"    {pct*100:.0f}% of weight in top {n_needed}/{N} channels")

    # Channel balance
    ch_norms = pc_w_sq.abs().sum(dim=1)  # L1 norm per output channel
    if policy_channels >= 2:
        ratio = ch_norms.max().item() / max(ch_norms.min().item(), 1e-8)
        balance_str = "  ".join(f"ch{i}_L1={ch_norms[i].item():.4f}" for i in range(policy_channels))
        print(f"\n  Channel balance: {balance_str}  ratio={ratio:.2f}x")
        # Correlation between channels' backbone usage
        corr = np.corrcoef(pc_w_sq[0].numpy(), pc_w_sq[1].numpy())[0, 1]
        print(f"  Inter-channel correlation: {corr:+.4f}")
        if corr > 0.8:
            print(f"  WARNING: High correlation -- channels may be redundant")
        elif corr < -0.5:
            print(f"  OK: Negative correlation -- channels capture different features")
    else:
        print(f"\n  Single policy channel: L1={ch_norms[0].item():.4f}")

    # --- 7b: Policy norm analysis ---
    p_bn_gamma = sd["policy_bn.weight"]
    p_bn_beta = sd["policy_bn.bias"]
    p_bn_var = sd.get("policy_bn.running_var")
    p_bn_mean = sd.get("policy_bn.running_mean")

    p_norm_type = "BN" if p_bn_var is not None else "GN"
    print(f"\n  Policy {p_norm_type} per channel:")
    if p_bn_var is not None:
        print(f"  {'ch':>3} | {'gamma':>7} | {'beta':>7} | {'var':>8} | {'mean':>8} | {'eff_gain':>8}")
        for i in range(p_bn_gamma.shape[0]):
            eff_gain = p_bn_gamma[i].item() / (p_bn_var[i].item() + 1e-5)**0.5
            print(f"  {i:>3} | {p_bn_gamma[i].item():>7.4f} | {p_bn_beta[i].item():>7.4f} | {p_bn_var[i].item():>8.6f} | {p_bn_mean[i].item():>8.6f} | {eff_gain:>8.2f}")
        eff_gains = p_bn_gamma / (p_bn_var + 1e-5).sqrt()
        near_zero = (eff_gains.abs() < 0.1).sum().item()
        if near_zero > 0:
            print(f"  WARNING: {near_zero}/{p_bn_gamma.shape[0]} channels near-zero effective gain")
    else:
        print(f"  {'ch':>3} | {'gamma':>7} | {'beta':>7}")
        for i in range(p_bn_gamma.shape[0]):
            print(f"  {i:>3} | {p_bn_gamma[i].item():>7.4f} | {p_bn_beta[i].item():>7.4f}")
        near_zero = (p_bn_gamma.abs() < 0.1).sum().item()
        if near_zero > 0:
            print(f"  WARNING: {near_zero}/{p_bn_gamma.shape[0]} channels near-zero gamma")

    # --- 7c: Policy FC analysis ---
    pfc_w = sd["policy_fc.weight"]  # shape: [action_size, 2*board_area]
    pfc_b = sd["policy_fc.bias"]    # shape: [action_size]
    in_features = pfc_w.shape[1]

    print(f"\n  Policy FC weight shape: {list(pfc_w.shape)}")
    print(f"  (maps {in_features} features -> {action_size} actions)")

    # Per-action (row) analysis
    row_norms = pfc_w.norm(dim=1)  # L2 norm per action
    row_l1 = pfc_w.abs().sum(dim=1)
    sorted_actions = row_norms.argsort(descending=True)

    print(f"\n  Per-action weight norms:")
    print(f"    L2: min={row_norms.min().item():.4f}  p25={row_norms.quantile(0.25).item():.4f}  "
          f"p50={row_norms.quantile(0.5).item():.4f}  p75={row_norms.quantile(0.75).item():.4f}  max={row_norms.max().item():.4f}")
    print(f"    Ratio max/min: {row_norms.max().item() / max(row_norms.min().item(), 1e-8):.2f}x")

    # Dead actions (very low weight norm)
    dead_thresh = row_norms.max().item() * 0.01
    weak_thresh = row_norms.max().item() * 0.1
    dead_actions = (row_norms < dead_thresh).sum().item()
    weak_actions = (row_norms < weak_thresh).sum().item()
    print(f"    Dead actions (<1% of max norm): {dead_actions}/{action_size}")
    print(f"    Weak actions (<10% of max norm): {weak_actions}/{action_size}")

    print(f"\n  Strongest 5 actions: {sorted_actions[:5].tolist()}")
    print(f"  Weakest 5 actions:   {sorted_actions[-5:].flip(0).tolist()}")

    # Per-input-feature (column) analysis -- which spatial positions matter
    col_norms = pfc_w.norm(dim=0)  # L2 norm per input feature
    print(f"\n  Per-input-feature norms (which spatial features matter):")
    print(f"    Total features: {in_features} (= {policy_channels} channels x {board_area} positions)")
    # Split into policy channels
    for ch_idx in range(policy_channels):
        ch_norms_slice = col_norms[ch_idx * board_area : (ch_idx + 1) * board_area]
        print(f"    Ch{ch_idx} positions: mean={ch_norms_slice.mean().item():.4f}  sum={ch_norms_slice.sum().item():.4f}")
    if policy_channels >= 2:
        ch0_sum = col_norms[:board_area].sum().item()
        ch1_sum = col_norms[board_area:2*board_area].sum().item()
        print(f"    Channel balance (ch0/ch1): {ch0_sum / max(ch1_sum, 1e-8):.2f}x")

    # SVD effective rank of policy FC
    svs = torch.linalg.svdvals(pfc_w)
    energy = (svs**2).cumsum(0) / (svs**2).sum()
    max_rank = min(action_size, in_features)
    rank_90 = (energy <= 0.9).sum().item() + 1
    rank_99 = (energy <= 0.99).sum().item() + 1
    near_zero_sv = (svs < 1e-6).sum().item()
    print(f"\n  Policy FC SVD analysis:")
    print(f"    Matrix: {action_size}x{in_features} (max rank = {max_rank})")
    print(f"    rank_90: {rank_90}/{max_rank}  rank_99: {rank_99}/{max_rank}  near_zero_sv: {near_zero_sv}")
    print(f"    sv[0]={svs[0].item():.4f}  sv[{max_rank//2}]={svs[max_rank//2].item():.6f}  sv[-1]={svs[-1].item():.2e}")

    # Capacity analysis
    print(f"\n  Capacity analysis:")
    print(f"    Input features: {in_features}  Output actions: {action_size}")
    if in_features < action_size:
        print(f"    WARNING: BOTTLENECK: {in_features} features -> {action_size} actions ({action_size/in_features:.1f}x expansion)")
        print(f"      Each action can only use {in_features} degrees of freedom")
        print(f"      Effective rank {rank_90} suggests {rank_90} independent action patterns")
    elif in_features > action_size * 2:
        print(f"    WARNING: OVERCAPACITY: {in_features} features -> {action_size} actions ({in_features/action_size:.1f}x compression)")
    else:
        print(f"    OK: Balanced: {in_features} features -> {action_size} actions")

    # Bias analysis
    bias_sorted = pfc_b.argsort(descending=True)
    print(f"\n  Policy FC bias:")
    print(f"    Range: [{pfc_b.min().item():+.4f}, {pfc_b.max().item():+.4f}]  mean={pfc_b.mean().item():+.4f}")
    print(f"    Most positive: action {bias_sorted[0].item()} = {pfc_b[bias_sorted[0]].item():+.4f}")
    print(f"    Most negative: action {bias_sorted[-1].item()} = {pfc_b[bias_sorted[-1]].item():+.4f}")

    return row_norms, svs


def section_summary(sd, composite, value_per_bb, policy_per_bb, importance, imp_sorted, arch):
    """Section 8: Summary statistics for interpretation."""
    N = arch["num_filters"]

    print("\n" + "="*80)
    print("SECTION 8: SUMMARY & INTERPRETATION EVIDENCE")
    print("="*80)

    really_dead = (composite < composite.quantile(0.05)).sum().item()
    print(f"\n  Backbone channels with composite norm < p5: {really_dead} (truly dead)")
    print(f"  Backbone channels with composite norm < p25: {(composite < composite.quantile(0.25)).sum().item()} (weak)")

    topK = min(20, N)
    v_topK = value_per_bb.argsort(descending=True)[:topK]
    v_topK_health = composite[v_topK].mean().item()
    all_health = composite.mean().item()
    p_topK = policy_per_bb.argsort(descending=True)[:topK]
    p_topK_health = composite[p_topK].mean().item()

    print(f"\n  Mean backbone health (all): {all_health:.4f}")
    print(f"  Mean backbone health (value top-{topK}): {v_topK_health:.4f}  ({v_topK_health/all_health:.2f}x)")
    print(f"  Mean backbone health (policy top-{topK}): {p_topK_health:.4f}  ({p_topK_health/all_health:.2f}x)")

    top4_imp = importance[imp_sorted[:4]].sum().item()
    total_imp = importance.sum().item()
    print(f"\n  FC1 top-4 neurons carry {top4_imp/total_imp*100:.1f}% of total importance")
    print(f"  FC1 neurons with importance > 1% of max: {(importance > importance.max()*0.01).sum().item()}")


def main():
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = find_latest_checkpoint()

    sd = load_checkpoint(ckpt_path)
    arch = detect_architecture(sd)

    print(f"\nArchitecture: {arch['num_filters']} filters, {arch['num_res_blocks']} res blocks, "
          f"{arch['value_head_channels']} value channels, {arch['value_head_fc_size']} fc size, "
          f"board_area={arch['board_area']}, actions={arch['action_size']}")
    print(f"Total parameters: {sum(p.numel() for p in sd.values()):,}")

    composite, _, _ = section_backbone_health(sd, arch)
    value_per_bb, policy_per_bb = section_policy_vs_value(sd, composite)
    importance, imp_sorted = section_fc1_analysis(sd)
    section_value_conv(sd)
    section_svd_rank(sd, arch)
    section_bn_analysis(sd, arch)
    section_policy_head(sd, arch)
    section_summary(sd, composite, value_per_bb, policy_per_bb, importance, imp_sorted, arch)

    print("\nDone.")


if __name__ == "__main__":
    main()
