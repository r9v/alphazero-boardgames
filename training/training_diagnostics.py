"""Training diagnostics collection and aggregation.

Extracted from Trainer to keep the training loop focused on optimization
while diagnostics logic lives here. All functions accept explicit data
arguments rather than reaching into trainer internals.
"""

import numpy as np
import torch
import torch.nn.functional as F

from utils import wdl_to_scalar


def make_accumulator():
    """Create the tracking accumulator dict used during a training pass."""
    return {
        'total_loss': 0, 'total_value_loss': 0, 'total_policy_loss': 0,
        'num_batches': 0, 'data_prep_time': 0.0, 'gradient_time': 0.0,
        'early_vloss': 0.0, 'early_ploss': 0.0,
        'late_vloss': 0.0, 'late_ploss': 0.0,
        'all_pred_vs': [],
        'value_grad_norms': [], 'policy_grad_norms': [],
        'x_vloss_sum': 0.0, 'o_vloss_sum': 0.0,
        'x_count': 0, 'o_count': 0,
        'x_target_sum': 0.0, 'o_target_sum': 0.0,
        'x_pred_sum': 0.0, 'o_pred_sum': 0.0,
        'all_policy_entropy': [],
        'top1_correct_sum': 0, 'top3_correct_sum': 0, 'policy_acc_count': 0,
        'confident_correct_sum': 0, 'confident_total': 0,
        'all_rb_grad_norms': {},
        'sub_iter_log': [],
        'conf_buckets': {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0},
        'conf_total': 0,
        'phase_vloss_sums': {'early': 0.0, 'mid': 0.0, 'late': 0.0},
        'phase_counts': {'early': 0, 'mid': 0, 'late': 0},
        'ploss_decisive_sum': 0.0, 'ploss_ambiguous_sum': 0.0,
        'decisive_count': 0, 'ambiguous_count': 0,
    }


def collect_step_diagnostics(step, states, target_vs, target_pis,
                             pred_vs, pred_pi_logits, value_loss, policy_loss,
                             acc, cfg, net, value_params, policy_params,
                             grad_stats_list, device):
    """Sample diagnostics at various intervals during training. Mutates acc.

    Args:
        step: Current training step index.
        states: Batch of input states tensor.
        target_vs: Target value class indices tensor.
        target_pis: Target policy distribution tensor.
        pred_vs: Predicted WDL logits tensor.
        pred_pi_logits: Predicted policy logits tensor.
        value_loss: Scalar value loss for this step.
        policy_loss: Scalar policy loss for this step.
        acc: Accumulator dict (mutated in place).
        cfg: Dict with 'num_steps', 'early_cutoff', 'late_start'.
        net: The neural network (for accessing specific layers/blocks).
        value_params: List of value head parameters.
        policy_params: List of policy head parameters.
        grad_stats_list: Mutable list to accumulate gradient statistics.
        device: Torch device.
    """
    num_steps = cfg['num_steps']
    early_cutoff = cfg['early_cutoff']
    late_start = cfg['late_start']

    # Cache wdl_to_scalar: compute once per step, reuse across all blocks
    _scalar_v = None
    _scalar_v_np = None
    needs_scalar = (step % 10 == 0) or (step % 50 == 0) or (step % 100 == 0) or (step >= late_start)
    if needs_scalar:
        with torch.no_grad():
            _scalar_v = wdl_to_scalar(pred_vs.detach())
        if step >= late_start:
            _scalar_v_np = _scalar_v.cpu().numpy()

    # (A) Per-player loss breakdown every 10 steps
    if step % 10 == 0:
        _collect_per_player_stats(states, target_vs, pred_vs, _scalar_v, acc)

    # (F) Gradient stats every 50 steps
    if step % 50 == 0:
        _collect_gradient_stats(target_vs, _scalar_v, net, grad_stats_list)

    # Sample gradient norms every 100 steps
    if step % 100 == 0:
        _collect_grad_norms(step, value_loss, policy_loss, _scalar_v,
                            acc, net, value_params, policy_params, device)

    # Track early vs late loss
    if step < early_cutoff:
        acc['early_vloss'] += value_loss.item()
        acc['early_ploss'] += policy_loss.item()
    if step >= late_start:
        acc['late_vloss'] += value_loss.item()
        acc['late_ploss'] += policy_loss.item()

    # Sample predictions from last 10% of steps for distribution analysis
    if step >= late_start:
        _collect_late_stage_metrics(
            target_vs, target_pis, pred_vs, pred_pi_logits,
            _scalar_v, _scalar_v_np, acc)


def _collect_per_player_stats(states, target_vs, pred_vs, scalar_v, acc):
    """Per-player loss breakdown and game-phase value loss."""
    with torch.no_grad():
        my_counts = states[:, 0].sum(dim=(1, 2))
        opp_counts = states[:, 1].sum(dim=(1, 2))
        is_x = (my_counts == opp_counts)
        is_o = ~is_x

        per_sample_vloss = F.cross_entropy(pred_vs, target_vs, reduction='none')
        scalar_target = (1 - target_vs.float())
        if is_x.any():
            acc['x_vloss_sum'] += per_sample_vloss[is_x].mean().item()
            acc['x_target_sum'] += scalar_target[is_x].mean().item()
            acc['x_pred_sum'] += scalar_v[is_x].mean().item()
            acc['x_count'] += 1
        if is_o.any():
            acc['o_vloss_sum'] += per_sample_vloss[is_o].mean().item()
            acc['o_target_sum'] += scalar_target[is_o].mean().item()
            acc['o_pred_sum'] += scalar_v[is_o].mean().item()
            acc['o_count'] += 1

        # Value loss by game phase
        total_pieces = my_counts + opp_counts
        for phase, lo, hi in [('early', 0, 8), ('mid', 9, 20), ('late', 21, 999)]:
            mask = (total_pieces >= lo) & (total_pieces <= hi)
            if mask.any():
                acc['phase_vloss_sums'][phase] += per_sample_vloss[mask].mean().item()
                acc['phase_counts'][phase] += 1


def _collect_gradient_stats(target_vs, scalar_v, net, grad_stats_list):
    """Collect value head gradient statistics."""
    with torch.no_grad():
        st = 1 - target_vs.float()
        error_scalar = scalar_v - st
        fc1_grad = net.value_fc1.weight.grad
        fc2_grad = net.value_fc2.weight.grad
        if fc1_grad is not None and fc2_grad is not None:
            fc1_per_neuron_gnorm = fc1_grad.norm(dim=1).cpu().numpy()
            vconv_g = net.value_conv.weight.grad
            vbn_g = net.value_bn.weight.grad
            grad_stats_list.append({
                'fc1_grad_mean': fc1_grad.mean().item(),
                'fc1_grad_std': fc1_grad.std().item(),
                'fc1_grad_norm': fc1_grad.norm().item(),
                'fc2_grad_mean': fc2_grad.mean().item(),
                'fc2_grad_std': fc2_grad.std().item(),
                'fc2_grad_norm': fc2_grad.norm().item(),
                'pred_mean': scalar_v.mean().item(),
                'target_mean': st.mean().item(),
                'error_mean': error_scalar.mean().item(),
                'fc1_per_neuron_gnorm': fc1_per_neuron_gnorm,
                'vconv_grad_norm': vconv_g.norm().item() if vconv_g is not None else 0.0,
                'vbn_gamma_grad_norm': vbn_g.norm().item() if vbn_g is not None else 0.0,
            })


def _collect_grad_norms(step, value_loss, policy_loss, scalar_v,
                        acc, net, value_params, policy_params, device):
    """Collect gradient norms for value/policy heads and residual blocks."""
    v_norm = sum(p.grad.norm().item() ** 2
                 for p in value_params if p.grad is not None) ** 0.5
    p_norm = sum(p.grad.norm().item() ** 2
                 for p in policy_params if p.grad is not None) ** 0.5
    acc['value_grad_norms'].append(v_norm)
    acc['policy_grad_norms'].append(p_norm)

    # Per-residual-block gradient norms
    for i, block in enumerate(net.res_blocks):
        rb_norm = sum(
            p.grad.norm().item() ** 2
            for p in block.parameters() if p.grad is not None
        ) ** 0.5
        acc['all_rb_grad_norms'].setdefault(i, []).append(rb_norm)

        c2w = block.conv2.weight
        if c2w.grad is not None:
            c2_eff_lr = c2w.grad.norm().item() / max(c2w.data.norm().item(), 1e-8)
            acc['all_rb_grad_norms'].setdefault(f'{i}_eff_lr', []).append(c2_eff_lr)

    # Sub-iteration logging
    acc['sub_iter_log'].append({
        'step': step,
        'vloss': value_loss.item(),
        'ploss': policy_loss.item(),
        'v_grad': acc['value_grad_norms'][-1] if acc['value_grad_norms'] else 0,
        'p_grad': acc['policy_grad_norms'][-1] if acc['policy_grad_norms'] else 0,
        'mean_conf': scalar_v.abs().mean().item(),
        'mean_v': scalar_v.mean().item(),
    })


def _collect_late_stage_metrics(target_vs, target_pis, pred_vs, pred_pi_logits,
                                scalar_v, scalar_v_np, acc):
    """Collect metrics from the last 10% of training steps."""
    acc['all_pred_vs'].append(scalar_v_np)

    # Value confidence distribution buckets
    _abs_v = np.abs(scalar_v_np)
    acc['conf_total'] += len(_abs_v)
    acc['conf_buckets']['very_low'] += int((_abs_v < 0.1).sum())
    acc['conf_buckets']['low'] += int(((_abs_v >= 0.1) & (_abs_v < 0.3)).sum())
    acc['conf_buckets']['medium'] += int(((_abs_v >= 0.3) & (_abs_v < 0.6)).sum())
    acc['conf_buckets']['high'] += int(((_abs_v >= 0.6) & (_abs_v < 0.9)).sum())
    acc['conf_buckets']['very_high'] += int((_abs_v >= 0.9).sum())

    # Policy quality metrics
    with torch.no_grad():
        pred_pis = F.softmax(pred_pi_logits.detach(), dim=1)
        log_pi = F.log_softmax(pred_pi_logits.detach(), dim=1)
        batch_entropy = -(pred_pis * log_pi).sum(dim=1).mean().item()
        acc['all_policy_entropy'].append(batch_entropy)

        pred_top = pred_pi_logits.argmax(dim=1)
        target_top = target_pis.argmax(dim=1)
        acc['top1_correct_sum'] += (pred_top == target_top).float().sum().item()

        pred_top3 = pred_pi_logits.topk(3, dim=1).indices
        target_argmax = target_pis.argmax(dim=1).unsqueeze(1)
        acc['top3_correct_sum'] += (pred_top3 == target_argmax).any(dim=1).float().sum().item()

        acc['policy_acc_count'] += pred_pis.shape[0]

        # Value confidence calibration
        scalar_tgt = 1 - target_vs.float()
        confident_mask = scalar_v.abs() > 0.5
        if confident_mask.any():
            confident_signs_correct = (
                scalar_v[confident_mask].sign() == scalar_tgt[confident_mask].sign()
            ).float()
            acc['confident_correct_sum'] += confident_signs_correct.sum().item()
            acc['confident_total'] += confident_mask.sum().item()

        # Policy loss on decisive vs ambiguous positions
        decisive_mask = (target_vs != 1)
        ambig_mask = (target_vs == 1)
        per_sample_ploss = -torch.sum(target_pis * log_pi, dim=1)
        if decisive_mask.any():
            acc['ploss_decisive_sum'] += per_sample_ploss[decisive_mask].mean().item()
            acc['decisive_count'] += 1
        if ambig_mask.any():
            acc['ploss_ambiguous_sum'] += per_sample_ploss[ambig_mask].mean().item()
            acc['ambiguous_count'] += 1


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_validation_loss(net, val_samples, batch_size, device, use_amp):
    """Compute held-out validation value and policy loss.

    Returns:
        (val_vloss, val_ploss) averaged over validation batches.
    """
    from training.trainer import raw_value_to_wdl_class

    val_vloss = 0.0
    val_ploss = 0.0
    net.eval()
    with torch.no_grad():
        val_batches = 0
        for i in range(0, len(val_samples), batch_size):
            vb = val_samples[i:i + batch_size]
            if len(vb) < 2:
                continue
            vs = torch.FloatTensor(np.array([s[0] for s in vb])).to(device)
            vt_pi = torch.FloatTensor(np.array([s[1] for s in vb])).to(device)
            vt_raw = np.array([s[2] for s in vb])
            vt_v = torch.LongTensor(raw_value_to_wdl_class(vt_raw)).to(device)
            pv, pp_logits = net(vs)[:2]
            val_vloss += F.cross_entropy(pv, vt_v).item()
            val_ploss += -torch.mean(torch.sum(vt_pi * F.log_softmax(pp_logits, dim=1), dim=1)).item()
            val_batches += 1
        if val_batches > 0:
            val_vloss /= val_batches
            val_ploss /= val_batches
    net.train()
    return val_vloss, val_ploss


def compute_svd_info(net):
    """Compute SVD rank of last resblock conv2 (capacity indicator).

    Returns:
        Dict with 'rank90', 'rank99', 'n_filters', or empty dict on failure.
    """
    try:
        last_rb = net.res_blocks[-1]
        w = last_rb.conv2.weight.data.reshape(last_rb.conv2.weight.shape[0], -1)
        sv = torch.linalg.svdvals(w)
        sv_norm = sv / sv[0]
        n_filters = w.shape[0]
        return {
            'rank90': int((sv_norm > 0.1).sum().item()),
            'rank99': int((sv_norm > 0.01).sum().item()),
            'n_filters': n_filters,
        }
    except Exception:
        return {}


def compute_backbone_drift(pre_train_backbone, net):
    """Compute cosine similarity between pre- and post-training backbone params.

    Returns:
        Float cosine similarity, or None on failure.
    """
    try:
        post_params = torch.cat([p.data.flatten() for p in net.res_blocks.parameters()])
        cos = F.cosine_similarity(pre_train_backbone.unsqueeze(0),
                                  post_params.unsqueeze(0)).item()
        return cos
    except Exception:
        return None


def summarize_grad_stats(grad_stats_list):
    """Summarize collected gradient stats into averages.

    Args:
        grad_stats_list: List of per-step gradient stat dicts.

    Returns:
        Summary dict, or empty dict if no stats collected.
    """
    if not grad_stats_list:
        return {}
    gs = grad_stats_list
    all_per_neuron = np.stack([g['fc1_per_neuron_gnorm'] for g in gs])
    avg_per_neuron_gnorm = all_per_neuron.mean(axis=0)
    return {
        'fc1_grad_norm_mean': np.mean([g['fc1_grad_norm'] for g in gs]),
        'fc2_grad_norm_mean': np.mean([g['fc2_grad_norm'] for g in gs]),
        'fc1_grad_mean': np.mean([g['fc1_grad_mean'] for g in gs]),
        'fc2_grad_mean': np.mean([g['fc2_grad_mean'] for g in gs]),
        'error_mean_trend': [g['error_mean'] for g in gs],
        'fc1_per_neuron_gnorm': avg_per_neuron_gnorm,
        'vconv_grad_norm': np.mean([g['vconv_grad_norm'] for g in gs]),
        'vbn_gamma_grad_norm': np.mean([g['vbn_gamma_grad_norm'] for g in gs]),
    }


def aggregate_training_results(acc, val_samples, cfg, setup,
                                net, batch_size, device, use_amp,
                                buffer_fill, buffer_max_size,
                                pre_train_backbone, grad_stats_list):
    """Compute averages, run validation, and build diagnostics dicts.

    Args:
        acc: Accumulator dict from the training loop.
        val_samples: Held-out validation samples.
        cfg: Training config dict with 'early_cutoff', etc.
        setup: Setup dict with 'n_samples', etc.
        net: The neural network.
        batch_size: Training batch size.
        device: Torch device string.
        use_amp: Whether AMP is enabled.
        buffer_fill: Current buffer occupancy.
        buffer_max_size: Buffer capacity.
        pre_train_backbone: Snapshotted backbone params tensor (or None).
        grad_stats_list: List of gradient stat dicts collected during training.

    Returns:
        (train_losses, train_diag, train_perf) where:
        - train_losses: (avg_loss, avg_value_loss, avg_policy_loss)
        - train_diag: Dict of diagnostic metrics for logging.
        - train_perf: Dict of performance timing metrics.
    """
    num_batches = acc['num_batches']
    early_cutoff = cfg['early_cutoff']

    avg_loss = acc['total_loss'] / max(num_batches, 1)
    avg_value_loss = acc['total_value_loss'] / max(num_batches, 1)
    avg_policy_loss = acc['total_policy_loss'] / max(num_batches, 1)

    # Value prediction distribution (from last 10% of training)
    all_pred_vs = acc['all_pred_vs']
    pred_v_all = np.concatenate(all_pred_vs) if all_pred_vs else np.array([0.0])
    pred_v_std = pred_v_all.std()

    # Policy quality metrics
    policy_top1_acc = acc['top1_correct_sum'] / max(acc['policy_acc_count'], 1)

    # Per-block gradient norms
    avg_rb_grad_norms = {}
    for i, norms in acc['all_rb_grad_norms'].items():
        avg_rb_grad_norms[i] = float(np.mean(norms))

    # Held-out validation loss
    val_vloss, val_ploss = compute_validation_loss(
        net, val_samples, batch_size, device, use_amp)

    # SVD rank
    svd_info = compute_svd_info(net)

    # Backbone drift
    drift_cos = None
    if pre_train_backbone is not None:
        drift_cos = compute_backbone_drift(pre_train_backbone, net)

    # Game phase value loss
    gp = {}
    for phase in ('early', 'mid', 'late'):
        c = acc['phase_counts'][phase]
        if c > 0:
            gp[phase] = acc['phase_vloss_sums'][phase] / c

    # Gradient stats summary
    grad_stats_summary = summarize_grad_stats(grad_stats_list)

    train_diag = {
        "avg_value_loss": avg_value_loss,
        "val_vloss": val_vloss,
        "buffer_fill": buffer_fill,
        "buffer_capacity": buffer_max_size,
        "pred_v_std": pred_v_std,
        "policy_top1_acc": policy_top1_acc,
        "rb_grad_norms": avg_rb_grad_norms,
        "svd": svd_info,
        "drift_cos": drift_cos,
        "game_phase_vloss": gp,
        "grad_stats_summary": grad_stats_summary,
    }

    train_perf = {
        "data_prep_time": acc['data_prep_time'],
        "gradient_time": acc['gradient_time'],
        "num_samples": setup['n_samples'],
        "num_batches": num_batches,
    }

    train_losses = (avg_loss, avg_value_loss, avg_policy_loss)
    return train_losses, train_diag, train_perf
