"""Extracted diagnostic computations for the training loop.

These functions were extracted from Trainer.train_network() and Trainer.run()
to reduce the size of those methods from ~1400 and ~900 lines respectively.
All diagnostic logic is preserved exactly; only the organization changed.
"""
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F

from utils import wdl_to_scalar

logger = logging.getLogger(__name__)


def raw_value_to_wdl_class(raw_v):
    """Convert raw values (+1/0/-1) to WDL class indices (0/1/2).

    Maps: +1 -> class 0 (win), 0 -> class 1 (draw), -1 -> class 2 (loss).
    """
    return (1 - raw_v).astype(np.int64)


def compute_three_in_a_row(inputs, values, sign_split=False):
    """Scan for positions with 3+ consecutive pieces on ch0 vs ch1.

    Args:
        inputs: np.array of shape (N, C, H, W)
        values: np.array of shape (N,) target values
        sign_split: if True, also compute sign distribution for ch0 3-in-a-row

    Returns:
        dict with 'mine' and 'opp' keys, each containing count/mean_target/frac.
        If sign_split=True, also includes 'sign_split' key with pos/neg/zero/n.
    """
    result = {}
    for ch, ch_name in [(0, 'mine'), (1, 'opp')]:
        ch_data = inputs[:, ch]
        h3 = ch_data[:, :, :-2] * ch_data[:, :, 1:-1] * ch_data[:, :, 2:]
        has_h3 = h3.any(axis=(1, 2))
        v3 = ch_data[:, :-2, :] * ch_data[:, 1:-1, :] * ch_data[:, 2:, :]
        has_v3 = v3.any(axis=(1, 2))
        mask = has_h3 | has_v3
        count = int(mask.sum())
        mean_target = float(values[mask].mean()) if count > 0 else 0.0
        result[ch_name] = {
            'count': count, 'mean_target': mean_target,
            'frac': count / len(inputs),
        }
    if sign_split:
        ch0_data = inputs[:, 0]
        h3 = ch0_data[:, :, :-2] * ch0_data[:, :, 1:-1] * ch0_data[:, :, 2:]
        v3 = ch0_data[:, :-2, :] * ch0_data[:, 1:-1, :] * ch0_data[:, 2:, :]
        ch0_mask = h3.any(axis=(1, 2)) | v3.any(axis=(1, 2))
        if ch0_mask.any():
            _tgts = values[ch0_mask]
            result['sign_split'] = {
                'pos': int((_tgts > 0).sum()),
                'neg': int((_tgts < 0).sum()),
                'zero': int((_tgts == 0).sum()),
                'n': len(_tgts),
            }
    return result


def compute_buffer_diagnostics(samples, all_values):
    """Value target distribution, per-player target bias, 3-in-a-row scanning.

    Returns dict with diagnostic values.
    """
    diag = {}
    diag['val_mean'] = all_values.mean()
    diag['val_std'] = all_values.std()
    diag['frac_pos'] = (all_values > 0).mean()
    diag['frac_neg'] = (all_values < 0).mean()
    diag['frac_draw'] = (all_values == 0).mean()
    diag['val_hist'] = [
        float(((all_values >= -1.0) & (all_values < -0.5)).mean()),
        float(((all_values >= -0.5) & (all_values < 0.0)).mean()),
        float((all_values == 0.0).mean()),
        float(((all_values > 0.0) & (all_values <= 0.5)).mean()),
        float(((all_values > 0.5) & (all_values <= 1.0)).mean()),
    ]

    # Per-player target bias
    diag['n_x_buf'] = diag['n_o_buf'] = 0
    diag['mean_tgt_x'] = diag['mean_tgt_o'] = 0.0
    diag['frac_pos_x'] = diag['frac_pos_o'] = 0.0
    try:
        # X moves when piece counts are equal (ch0==ch1), O when unequal
        is_x_buf = np.array([s[0][0].sum() == s[0][1].sum() for s in samples])
        is_o_buf = ~is_x_buf
        diag['n_x_buf'] = int(is_x_buf.sum())
        diag['n_o_buf'] = int(is_o_buf.sum())
        if diag['n_x_buf'] > 0:
            diag['mean_tgt_x'] = float(all_values[is_x_buf].mean())
            diag['frac_pos_x'] = float((all_values[is_x_buf] > 0).mean())
        if diag['n_o_buf'] > 0:
            diag['mean_tgt_o'] = float(all_values[is_o_buf].mean())
            diag['frac_pos_o'] = float((all_values[is_o_buf] > 0).mean())
    except (IndexError, ValueError) as e:
        logger.debug("Per-player target bias failed: %s", e)

    # 3-in-a-row target bias (sign_split=True gives ch0 sign distribution)
    diag['three_r_diag'] = {}
    try:
        all_inputs = np.array([s[0] for s in samples])
        three_r = compute_three_in_a_row(all_inputs, all_values, sign_split=True)
        diag['three_r_diag'] = three_r
        sign_split = three_r.get('sign_split')
        if sign_split:
            print(f"  Diag[SIGN]: ch0_3row: +:{sign_split['pos']} -:{sign_split['neg']} "
                  f"0:{sign_split['zero']} (n={sign_split['n']}, "
                  f"mean={three_r['mine']['mean_target']:+.4f})")
    except (IndexError, ValueError) as e:
        logger.debug("3-in-a-row diagnostic failed: %s", e)

    return diag


def compute_pre_training_diagnostics(net, samples, device):
    """Measure pre-training value loss and per-player predictions.

    Single forward pass computes both value loss and per-player bias.

    Returns:
        (pre_train_vloss, pbias_data) tuple.
    """
    pre_train_vloss = None
    pbias_data = None
    try:
        net.eval()
        with torch.no_grad():
            batch = random.choices(samples, k=min(512, len(samples)))
            states = torch.FloatTensor(np.array([s[0] for s in batch])).to(device)
            targets = np.array([s[2] for s in batch])
            targets_v = torch.LongTensor(raw_value_to_wdl_class(targets)).to(device)
            # Infer player from piece counts: equal = X to move
            is_x = np.array([s[0][0].sum() == s[0][1].sum() for s in batch])
            is_o = ~is_x

            pred_v, _ = net(states)

            # Value loss
            pre_train_vloss = float(F.cross_entropy(pred_v, targets_v).item())

            # Per-player bias
            v_scalar = wdl_to_scalar(pred_v).cpu().numpy()
            pbias_data = {
                'batch': batch, 'targets': targets,
                'is_x': is_x, 'is_o': is_o,
                'pre_x_pred': float(v_scalar[is_x].mean()) if is_x.any() else 0.0,
                'pre_o_pred': float(v_scalar[is_o].mean()) if is_o.any() else 0.0,
                'pre_x_acc': float(((v_scalar[is_x] > 0) == (targets[is_x] > 0)).mean()) if is_x.any() else 0.0,
                'pre_o_acc': float(((v_scalar[is_o] > 0) == (targets[is_o] > 0)).mean()) if is_o.any() else 0.0,
            }
        net.train()
    except (RuntimeError, ValueError) as e:
        logger.debug("Pre-training diagnostics failed: %s", e)

    return pre_train_vloss, pbias_data


def compute_post_training_player_diagnostics(net, pbias_data, device):
    """Per-player predictions after training.

    Returns:
        (post_x_pred, post_o_pred, post_x_acc, post_o_acc) tuple.
    """
    post_x_pred, post_o_pred = 0.0, 0.0
    post_x_acc, post_o_acc = 0.0, 0.0
    if pbias_data is not None:
        try:
            net.eval()
            with torch.no_grad():
                states = torch.FloatTensor(
                    np.array([s[0] for s in pbias_data['batch']])
                ).to(device)
                v, _ = net(states)
                v_scalar = wdl_to_scalar(v).cpu().numpy()
                targets = pbias_data['targets']
                is_x = pbias_data['is_x']
                is_o = pbias_data['is_o']
                if is_x.any():
                    post_x_pred = float(v_scalar[is_x].mean())
                    post_x_acc = float(((v_scalar[is_x] > 0) == (targets[is_x] > 0)).mean())
                if is_o.any():
                    post_o_pred = float(v_scalar[is_o].mean())
                    post_o_acc = float(((v_scalar[is_o] > 0) == (targets[is_o] > 0)).mean())
            net.train()
        except (RuntimeError, ValueError) as e:
            logger.debug("Post-training player diagnostics failed: %s", e)
    return post_x_pred, post_o_pred, post_x_acc, post_o_acc


def compute_value_head_diagnostics(trainer, samples, grad_stats_summary):
    """Value head health: dead neurons, WDL distribution, weight stats, backbone signal.

    Args:
        trainer: Trainer instance (for net, device, and _prev_* state)
        samples: training samples
        grad_stats_summary: gradient statistics from training loop

    Returns:
        vh_diag dict with all value head diagnostic metrics.
    """
    net = trainer.net
    device = trainer.device
    vh_diag = {}
    try:
        net.eval()
        captured = {}
        def hook_vconv(module, input, output):
            captured['backbone_raw'] = input[0]
            captured['vconv_out'] = output
        def hook_vbn(module, input, output):
            captured['vbn_out'] = output
        def hook_pconv(module, input, output):
            captured['pconv_out'] = output
        def hook_fc1(module, input, output):
            captured['fc1_in'] = input[0]
            captured['fc1_out'] = output
        def hook_fc2(module, input, output):
            captured['fc2_in'] = input[0]
            captured['wdl_logits'] = output
        def make_rb_hook(idx):
            def hook(module, input, output):
                captured[f'rb{idx}_in'] = input[0]
                captured[f'rb{idx}_out'] = output
            return hook
        rb_hooks = []
        for idx, block in enumerate(net.res_blocks):
            rb_hooks.append(block.register_forward_hook(make_rb_hook(idx)))
        def make_rb_bn2_hook(idx):
            def hook(module, input, output):
                captured[f'rb{idx}_bn2_in'] = input[0]
                captured[f'rb{idx}_bn2_out'] = output
            return hook
        def make_rb_conv2_hook(idx):
            def hook(module, input, output):
                captured[f'rb{idx}_conv2_out'] = output
            return hook
        rb_bn2_hooks = []
        for idx, block in enumerate(net.res_blocks):
            rb_bn2_hooks.append(block.bn2.register_forward_hook(make_rb_bn2_hook(idx)))
            rb_bn2_hooks.append(block.conv2.register_forward_hook(make_rb_conv2_hook(idx)))

        h0 = net.value_conv.register_forward_hook(hook_vconv)
        h0b = net.value_bn.register_forward_hook(hook_vbn)
        h0p = net.policy_conv.register_forward_hook(hook_pconv)
        h1 = net.value_fc1.register_forward_hook(hook_fc1)
        h2 = net.value_fc2.register_forward_hook(hook_fc2)
        with torch.no_grad():
            diag_batch = random.choices(samples, k=min(256, len(samples)))
            diag_inp = torch.FloatTensor(
                np.array([s[0] for s in diag_batch])
            ).to(device)
            v_out, _ = net(diag_inp)
        h0.remove(); h0b.remove(); h0p.remove(); h1.remove(); h2.remove()
        for h in rb_hooks:
            h.remove()
        for h in rb_bn2_hooks:
            h.remove()

        backbone_raw_np = captured['backbone_raw'].cpu().numpy()
        vconv_np = captured['vconv_out'].cpu().numpy()
        vbn_np = captured['vbn_out'].cpu().numpy()
        pconv_np = captured['pconv_out'].cpu().numpy()
        fc1_np = captured['fc1_out'].cpu().numpy()
        fc2_in_np = captured['fc2_in'].cpu().numpy()
        wdl_logits_np = captured['wdl_logits'].cpu().numpy()
        wdl_probs_diag = F.softmax(captured['wdl_logits'], dim=1).cpu().numpy()
        scalar_v_diag = wdl_probs_diag[:, 0] - wdl_probs_diag[:, 2]  # numpy, already probs
        fc1_in_np = captured['fc1_in'].cpu().numpy()

        wdl_entropy_per_sample = -(wdl_probs_diag * np.log(wdl_probs_diag + 1e-8)).sum(axis=1)
        wdl_confidence = wdl_probs_diag.max(axis=1)
        diag_raw_targets = np.array([s[2] for s in diag_batch])
        diag_target_classes = raw_value_to_wdl_class(diag_raw_targets)
        wdl_pred_classes = wdl_logits_np.argmax(axis=1)
        wdl_accuracy = float((wdl_pred_classes == diag_target_classes).mean())

        n_total_neurons = fc1_np.shape[1]
        near_dead = (np.abs(fc1_np) < 0.01).all(axis=0)
        n_dead = int(near_dead.sum())

        fc2_w = net.value_fc2.weight.detach().cpu().numpy()
        fc2_b = net.value_fc2.bias.detach().cpu().numpy()
        fc1_w = net.value_fc1.weight.detach().cpu().numpy()
        neuron_abs_mean = np.abs(fc2_in_np).mean(axis=0)
        active_count = int((neuron_abs_mean > 0.1).sum())

        fc1_abs = np.abs(fc1_np)
        fc1_abs_flat = fc1_abs.flatten()
        fc1_p10 = float(np.percentile(fc1_abs_flat, 10))
        fc1_p50 = float(np.percentile(fc1_abs_flat, 50))
        fc1_p90 = float(np.percentile(fc1_abs_flat, 90))

        n_vconv_ch = vconv_np.shape[1]
        vconv_ch_abs_mean = []
        vconv_ch_std = []
        vconv_dead_channels = 0
        for ch in range(n_vconv_ch):
            ch_data = vconv_np[:, ch, :, :]
            ch_abs = float(np.abs(ch_data).mean())
            ch_s = float(ch_data.std())
            vconv_ch_abs_mean.append(ch_abs)
            vconv_ch_std.append(ch_s)
            if ch_abs < 0.01:
                vconv_dead_channels += 1

        dead_neuron_ids = sorted(np.where(near_dead)[0].tolist())
        weakest_5_idx = np.argsort(neuron_abs_mean)[:5].tolist()
        weakest_5_vals = [float(neuron_abs_mean[i]) for i in weakest_5_idx]

        fc2_w_norm_now = float(np.linalg.norm(fc2_w))
        fc1_w_norm_now = float(np.linalg.norm(fc1_w))
        if not hasattr(trainer, '_prev_fc2_w_norm'):
            trainer._prev_fc2_w_norm = fc2_w_norm_now
            trainer._prev_fc1_w_norm = fc1_w_norm_now
        fc2_w_norm_delta = fc2_w_norm_now - trainer._prev_fc2_w_norm
        fc1_w_norm_delta = fc1_w_norm_now - trainer._prev_fc1_w_norm
        trainer._prev_fc2_w_norm = fc2_w_norm_now
        trainer._prev_fc1_w_norm = fc1_w_norm_now

        init_conv_w_norm = float(net.conv.weight.data.norm().item())
        init_conv_w_abs_mean = float(net.conv.weight.data.abs().mean().item())
        if not hasattr(trainer, '_prev_init_conv_w_norm'):
            trainer._prev_init_conv_w_norm = init_conv_w_norm
        init_conv_w_norm_delta = init_conv_w_norm - trainer._prev_init_conv_w_norm
        trainer._prev_init_conv_w_norm = init_conv_w_norm

        rb_conv_norms = {}
        if not hasattr(trainer, '_prev_rb_c2_norms'):
            trainer._prev_rb_c2_norms = {}
        for bi in range(len(net.res_blocks)):
            c1_norm = float(net.res_blocks[bi].conv1.weight.data.norm().item())
            c2_norm = float(net.res_blocks[bi].conv2.weight.data.norm().item())
            c2_delta = c2_norm - trainer._prev_rb_c2_norms.get(bi, c2_norm)
            trainer._prev_rb_c2_norms[bi] = c2_norm
            rb_conv_norms[bi] = {"conv1": c1_norm, "conv2": c2_norm, "c2_delta": c2_delta}

        grad_dead_mean = 0.0
        grad_alive_mean = 0.0
        gs_data = grad_stats_summary.get('fc1_per_neuron_gnorm')
        if gs_data is not None and len(gs_data) == n_total_neurons:
            dead_mask_arr = near_dead
            alive_mask_arr = ~dead_mask_arr
            if dead_mask_arr.any():
                grad_dead_mean = float(gs_data[dead_mask_arr].mean())
            if alive_mask_arr.any():
                grad_alive_mean = float(gs_data[alive_mask_arr].mean())

        backbone_raw_abs = float(np.abs(backbone_raw_np).mean())
        backbone_raw_std = float(backbone_raw_np.std())
        bb_ch_abs = np.abs(backbone_raw_np).mean(axis=(0, 2, 3))
        bb_n_channels = len(bb_ch_abs)
        bb_dead_channels = int((bb_ch_abs < 0.01).sum())
        bb_ch_p10 = float(np.percentile(bb_ch_abs, 10))
        bb_ch_p50 = float(np.percentile(bb_ch_abs, 50))
        bb_ch_p90 = float(np.percentile(bb_ch_abs, 90))
        bb_ch_max = float(bb_ch_abs.max())
        bb_top5_idx = np.argsort(bb_ch_abs)[-5:][::-1].tolist()
        bb_top5_vals = [float(bb_ch_abs[i]) for i in bb_top5_idx]
        bb_bot5_idx = np.argsort(bb_ch_abs)[:5].tolist()
        bb_bot5_vals = [float(bb_ch_abs[i]) for i in bb_bot5_idx]

        rb_act_stats = {}
        for idx in range(len(net.res_blocks)):
            key = f'rb{idx}_out'
            if key in captured:
                rb_np = captured[key].cpu().numpy()
                rb_abs = float(np.abs(rb_np).mean())
                rb_std_val = float(rb_np.std())
                rb_ch_abs = np.abs(rb_np).mean(axis=(0, 2, 3))
                rb_dead = int((rb_ch_abs < 0.01).sum())
                rb_act_stats[idx] = {
                    "abs_mean": rb_abs, "std": rb_std_val,
                    "dead_channels": rb_dead,
                    "ch_p50": float(np.percentile(rb_ch_abs, 50)),
                }

        rb_residual_ratios = {}
        for idx in range(len(net.res_blocks)):
            in_key = f'rb{idx}_in'
            out_key = f'rb{idx}_out'
            if in_key in captured and out_key in captured:
                rb_in = captured[in_key]
                rb_out = captured[out_key]
                residual = rb_out - rb_in
                skip_norm = float(rb_in.norm().item())
                res_norm = float(residual.norm().item())
                rb_residual_ratios[idx] = res_norm / max(skip_norm, 1e-10)

        rb_bn2_stats = {}
        for idx in range(len(net.res_blocks)):
            conv2_out_key = f'rb{idx}_conv2_out'
            bn2_out_key = f'rb{idx}_bn2_out'
            bn2_in_key = f'rb{idx}_bn2_in'
            stats = {}
            if conv2_out_key in captured:
                conv2_out = captured[conv2_out_key].cpu().numpy()
                stats["conv2_raw_var"] = float(conv2_out.var())
                stats["conv2_raw_abs"] = float(np.abs(conv2_out).mean())
            if bn2_out_key in captured:
                bn2_out = captured[bn2_out_key].cpu().numpy()
                stats["bn2_out_std"] = float(bn2_out.std())
                stats["bn2_out_abs"] = float(np.abs(bn2_out).mean())
            if bn2_in_key in captured:
                bn2_in_t = captured[bn2_in_key]
                batch_var = bn2_in_t.var(dim=(0, 2, 3))
                stats["bn2_batch_var_mean"] = float(batch_var.mean().item())
            if stats:
                rb_bn2_stats[idx] = stats

        rb_res_rank = {}
        for idx in range(len(net.res_blocks)):
            in_key = f'rb{idx}_in'
            out_key = f'rb{idx}_out'
            if in_key in captured and out_key in captured:
                residual = captured[out_key] - captured[in_key]
                r_mean = residual.mean(dim=0).flatten(1)
                svs = torch.linalg.svdvals(r_mean)
                energy = (svs**2).cumsum(0) / (svs**2).sum()
                rank90 = int((energy <= 0.9).sum().item()) + 1
                rank99 = int((energy <= 0.99).sum().item()) + 1
                rb_res_rank[idx] = {"rank90": rank90, "rank99": rank99, "total": r_mean.shape[0]}

        vconv_pre_bn_abs = float(np.abs(vconv_np).mean())
        vconv_pre_bn_std = float(vconv_np.std())
        vbn_abs = float(np.abs(vbn_np).mean())
        vbn_std = float(vbn_np.std())
        bn_ratio = vbn_abs / max(vconv_pre_bn_abs, 1e-10)
        vc_w = net.value_conv.weight.detach().cpu().numpy()
        vc_w_norm = float(np.linalg.norm(vc_w))
        vc_w_abs_mean = float(np.abs(vc_w).mean())
        vbn_gamma = net.value_bn.weight.detach().cpu().numpy()
        vbn_beta = net.value_bn.bias.detach().cpu().numpy()
        pconv_abs = float(np.abs(pconv_np).mean())
        pconv_std = float(pconv_np.std())
        vconv_grad_norm = grad_stats_summary.get('vconv_grad_norm', 0.0)

        vh_diag = {
            "dead_neurons": n_dead, "total_neurons": n_total_neurons,
            "active_neurons": active_count,
            "wdl_logit_std": float(wdl_logits_np.std()),
            "wdl_logit_range": float(wdl_logits_np.max() - wdl_logits_np.min()),
            "wdl_entropy": float(wdl_entropy_per_sample.mean()),
            "wdl_confidence": float(wdl_confidence.mean()),
            "wdl_win_prob": float(wdl_probs_diag[:, 0].mean()),
            "wdl_draw_prob": float(wdl_probs_diag[:, 1].mean()),
            "wdl_loss_prob": float(wdl_probs_diag[:, 2].mean()),
            "wdl_scalar_mean": float(scalar_v_diag.mean()),
            "wdl_scalar_std": float(scalar_v_diag.std()),
            "wdl_accuracy": wdl_accuracy,
            "fc2_w_max": float(fc2_w.max()), "fc2_w_min": float(fc2_w.min()),
            "fc2_w_norm": fc2_w_norm_now,
            "fc2_bias_w": float(fc2_b[0]), "fc2_bias_d": float(fc2_b[1]),
            "fc2_bias_l": float(fc2_b[2]),
            "fc1_w_norm": fc1_w_norm_now,
            "backbone_std": float(fc1_in_np.std()),
            "backbone_abs_mean": float(np.abs(fc1_in_np).mean()),
            "grad_dead_mean": grad_dead_mean, "grad_alive_mean": grad_alive_mean,
            "fc1_act_p10": fc1_p10, "fc1_act_p50": fc1_p50, "fc1_act_p90": fc1_p90,
            "vconv_dead_channels": vconv_dead_channels, "vconv_n_channels": n_vconv_ch,
            "vconv_ch_abs_mean": vconv_ch_abs_mean, "vconv_ch_std": vconv_ch_std,
            "dead_neuron_ids": dead_neuron_ids,
            "weakest_5_ids": weakest_5_idx, "weakest_5_vals": weakest_5_vals,
            "fc2_w_norm_delta": fc2_w_norm_delta, "fc1_w_norm_delta": fc1_w_norm_delta,
            "backbone_raw_abs": backbone_raw_abs, "backbone_raw_std": backbone_raw_std,
            "vconv_pre_bn_abs": vconv_pre_bn_abs, "vconv_pre_bn_std": vconv_pre_bn_std,
            "vbn_post_abs": vbn_abs, "vbn_post_std": vbn_std, "bn_ratio": bn_ratio,
            "vc_w_norm": vc_w_norm, "vc_w_abs_mean": vc_w_abs_mean,
            "vbn_gamma": vbn_gamma.tolist(), "vbn_beta": vbn_beta.tolist(),
            "vbn_gamma_mean": float(vbn_gamma.mean()),
            "vbn_gamma_min": float(vbn_gamma.min()),
            "pconv_abs": pconv_abs, "pconv_std": pconv_std,
            "vconv_grad_norm": vconv_grad_norm,
            "bb_n_channels": bb_n_channels, "bb_dead_channels": bb_dead_channels,
            "bb_ch_p10": bb_ch_p10, "bb_ch_p50": bb_ch_p50,
            "bb_ch_p90": bb_ch_p90, "bb_ch_max": bb_ch_max,
            "bb_top5": list(zip(bb_top5_idx, bb_top5_vals)),
            "bb_bot5": list(zip(bb_bot5_idx, bb_bot5_vals)),
            "rb_act_stats": rb_act_stats, "rb_residual_ratios": rb_residual_ratios,
            "init_conv_w_norm": init_conv_w_norm,
            "init_conv_w_abs_mean": init_conv_w_abs_mean,
            "init_conv_w_norm_delta": init_conv_w_norm_delta,
            "rb_conv_norms": rb_conv_norms, "rb_bn2_stats": rb_bn2_stats,
            "rb_res_rank": rb_res_rank,
        }
        net.train()
    except (RuntimeError, ValueError, IndexError, AttributeError) as e:
        print(f"  [DIAG-DBG] Value head diagnostic block failed: {e}")
    return vh_diag


def compute_backbone_gradient_decomposition(trainer, samples, vh_diag):
    """Separate backward passes to see which head drives each backbone channel.

    Updates vh_diag in place.
    """
    net = trainer.net
    device = trainer.device
    try:
        net.eval()
        gd_batch = random.choices(samples, k=min(64, len(samples)))
        gd_states = torch.FloatTensor(np.array([s[0] for s in gd_batch])).to(device)
        gd_raw_v = np.array([s[2] for s in gd_batch])
        gd_targets_v = torch.LongTensor(raw_value_to_wdl_class(gd_raw_v)).to(device)
        gd_targets_pi = torch.FloatTensor(np.array([s[1] for s in gd_batch])).to(device)

        bb_ref = {}
        def hook_bb_capture(module, inp, out):
            bb_ref['x'] = inp[0]
        h_bb = net.value_conv.register_forward_hook(hook_bb_capture)
        rb_gd_refs = {}
        def make_rb_gd_hook(idx):
            def hook(module, input, output):
                rb_gd_refs[f'rb{idx}'] = output
            return hook
        rb_gd_hooks = []
        for idx, block in enumerate(net.res_blocks):
            rb_gd_hooks.append(block.register_forward_hook(make_rb_gd_hook(idx)))

        trainer.optimizer.zero_grad()
        gd_pred_v, gd_pred_p_logits = net(gd_states)
        gd_v_loss = F.cross_entropy(gd_pred_v, gd_targets_v)
        gd_p_loss = -torch.mean(torch.sum(gd_targets_pi * F.log_softmax(gd_pred_p_logits, dim=1), dim=1))
        bb_x = bb_ref['x']
        h_bb.remove()

        v_grad_bb = torch.autograd.grad(gd_v_loss, bb_x, retain_graph=True)[0]
        p_grad_bb = torch.autograd.grad(gd_p_loss, bb_x, retain_graph=True)[0]
        v_grad_ch = v_grad_bb.abs().mean(dim=(0, 2, 3)).cpu().numpy()
        p_grad_ch = p_grad_bb.abs().mean(dim=(0, 2, 3)).cpu().numpy()
        v_grad_bb_norm = float(v_grad_bb.norm().item())
        p_grad_bb_norm = float(p_grad_bb.norm().item())

        bb_grad_cosine = float(F.cosine_similarity(
            v_grad_bb.flatten().unsqueeze(0),
            p_grad_bb.flatten().unsqueeze(0)
        ).item())
        v_ch_flat = v_grad_bb.mean(dim=0).flatten(1)
        p_ch_flat = p_grad_bb.mean(dim=0).flatten(1)
        ch_cosines = F.cosine_similarity(v_ch_flat, p_ch_flat, dim=1)
        bb_grad_conflict_channels = int((ch_cosines < 0).sum().item())
        bb_grad_aligned_channels = int((ch_cosines > 0.5).sum().item())

        ch_total = v_grad_ch + p_grad_ch + 1e-10
        ch_value_frac = v_grad_ch / ch_total
        n_value_dom = int((ch_value_frac > 0.5).sum())
        n_policy_dom = int((ch_value_frac <= 0.5).sum())
        top_v_ch = np.argsort(ch_value_frac)[-5:][::-1].tolist()
        top_p_ch = np.argsort(ch_value_frac)[:5].tolist()
        top_v_frac = [float(ch_value_frac[i]) for i in top_v_ch]
        top_p_frac = [float(ch_value_frac[i]) for i in top_p_ch]

        value_w = net.value_conv.weight.data
        policy_w = net.policy_conv.weight.data
        val_per_ch = value_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()
        pol_per_ch = policy_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()
        vp_corr = float(np.corrcoef(val_per_ch, pol_per_ch)[0, 1])
        top20_val = np.argsort(val_per_ch)[-20:]
        top20_pol = np.argsort(pol_per_ch)[-20:]
        vp_overlap_20 = len(set(top20_val.tolist()) & set(top20_pol.tolist()))
        bb_act_ch = bb_x.detach().abs().mean(dim=(0, 2, 3)).cpu().numpy()
        val_top20_act = float(bb_act_ch[top20_val].mean())
        pol_top20_act = float(bb_act_ch[top20_pol].mean())
        val_health_corr = float(np.corrcoef(val_per_ch, bb_act_ch)[0, 1])
        pol_health_corr = float(np.corrcoef(pol_per_ch, bb_act_ch)[0, 1])

        backbone_params = []
        for name, param in net.named_parameters():
            if not name.startswith('value') and not name.startswith('policy'):
                backbone_params.append((name, param))
        bp_list = [p for _, p in backbone_params]
        v_bp_grads = torch.autograd.grad(gd_v_loss, bp_list, retain_graph=True, allow_unused=True)
        p_bp_grads = torch.autograd.grad(gd_p_loss, bp_list, retain_graph=True, allow_unused=True)
        v_bp_sq = 0.0
        p_bp_sq = 0.0
        rb_v = {}
        rb_p = {}
        for (name, param), vg, pg in zip(backbone_params, v_bp_grads, p_bp_grads):
            vn = vg.norm().item() if vg is not None else 0.0
            pn = pg.norm().item() if pg is not None else 0.0
            v_bp_sq += vn ** 2
            p_bp_sq += pn ** 2
            if name.startswith('res_blocks.'):
                bn = name.split('.')[1]
                rb_v[bn] = rb_v.get(bn, 0.0) + vn ** 2
                rb_p[bn] = rb_p.get(bn, 0.0) + pn ** 2
        v_bp_total = v_bp_sq ** 0.5
        p_bp_total = p_bp_sq ** 0.5
        rb_summary = {}
        for bn in sorted(rb_v.keys()):
            rv = rb_v[bn] ** 0.5
            rp = rb_p.get(bn, 0.0) ** 0.5
            rb_summary[bn] = (rv, rp, rv / max(rp, 1e-10))

        rb_ch_dominance = {}
        for idx in range(len(net.res_blocks)):
            key = f'rb{idx}'
            if key in rb_gd_refs:
                rb_out = rb_gd_refs[key]
                v_grad_rb = torch.autograd.grad(gd_v_loss, rb_out, retain_graph=True)[0]
                p_grad_rb = torch.autograd.grad(gd_p_loss, rb_out, retain_graph=True)[0]
                v_ch = v_grad_rb.abs().mean(dim=(0, 2, 3)).cpu().numpy()
                p_ch = p_grad_rb.abs().mean(dim=(0, 2, 3)).cpu().numpy()
                total_ch = v_ch + p_ch + 1e-10
                v_frac = v_ch / total_ch
                rb_ch_dominance[idx] = int((v_frac > 0.5).sum())

        rb_v_grad_survival = {}
        for bn in sorted(rb_v.keys()):
            rv = rb_v[bn] ** 0.5
            rb_v_grad_survival[bn] = rv / max(v_grad_bb_norm, 1e-10)

        for h in rb_gd_hooks:
            h.remove()
        trainer.optimizer.zero_grad()
        net.train()

        vh_diag.update({
            "bb_v_grad_norm": v_grad_bb_norm, "bb_p_grad_norm": p_grad_bb_norm,
            "bb_grad_ratio": v_grad_bb_norm / max(p_grad_bb_norm, 1e-10),
            "bb_n_value_dom": n_value_dom, "bb_n_policy_dom": n_policy_dom,
            "bb_top_v_channels": list(zip(top_v_ch, top_v_frac)),
            "bb_top_p_channels": list(zip(top_p_ch, top_p_frac)),
            "bb_param_v_grad": v_bp_total, "bb_param_p_grad": p_bp_total,
            "bb_param_grad_ratio": v_bp_total / max(p_bp_total, 1e-10),
            "bb_res_block_grads": rb_summary,
            "vp_weight_corr": vp_corr, "vp_overlap_20": vp_overlap_20,
            "vp_val_top20_act": val_top20_act, "vp_pol_top20_act": pol_top20_act,
            "vp_val_health_corr": val_health_corr, "vp_pol_health_corr": pol_health_corr,
            "bb_grad_cosine_sim": bb_grad_cosine,
            "bb_grad_conflict_channels": bb_grad_conflict_channels,
            "bb_grad_aligned_channels": bb_grad_aligned_channels,
            "rb_ch_dominance": rb_ch_dominance,
            "rb_v_grad_survival": rb_v_grad_survival,
        })
    except (RuntimeError, ValueError, IndexError) as e:
        print(f"  [DIAG-DBG] Backbone gradient decomposition failed: {e}")


def compute_svd_rank_diagnostics(net, vh_diag):
    """SVD rank tracking & policy head internals.

    Updates vh_diag in place.
    """
    try:
        net.eval()
        rb_idx = len(net.res_blocks) - 1
        rb_w = net.res_blocks[rb_idx].conv2.weight.data.flatten(1)
        svs_bb = torch.linalg.svdvals(rb_w)
        energy_bb = (svs_bb**2).cumsum(0) / (svs_bb**2).sum()
        bb_rank90 = int((energy_bb <= 0.9).sum().item()) + 1
        bb_rank99 = int((energy_bb <= 0.99).sum().item()) + 1
        bb_near_zero_sv = int((svs_bb < 1e-6).sum().item())
        bb_n = rb_w.shape[0]

        # With GroupNorm, gamma directly scales output (no running_var).
        # Small |gamma| channels are effectively dead.
        gn_gamma = net.res_blocks[rb_idx].bn2.weight.data
        gn_dead = int((gn_gamma.abs() < 0.1).sum().item())

        pfc_w = net.policy_fc.weight.data
        svs_pfc = torch.linalg.svdvals(pfc_w)
        energy_pfc = (svs_pfc**2).cumsum(0) / (svs_pfc**2).sum()
        pfc_max_rank = min(pfc_w.shape[0], pfc_w.shape[1])
        pfc_rank90 = int((energy_pfc <= 0.9).sum().item()) + 1
        pfc_rank99 = int((energy_pfc <= 0.99).sum().item()) + 1

        pc_w = net.policy_conv.weight.data.squeeze(-1).squeeze(-1)
        if pc_w.dim() == 1:
            pc_w = pc_w.unsqueeze(0)
        pc_ch_norms = pc_w.abs().sum(dim=1).cpu().tolist()

        all_rb_bn = {}
        for bi in range(len(net.res_blocks)):
            rb_bn_gamma = net.res_blocks[bi].bn2.weight.data
            rb_neg_gamma = int((rb_bn_gamma < -0.01).sum().item())
            rb_gn_dead = int((rb_bn_gamma.abs() < 0.1).sum().item())
            all_rb_bn[bi] = {
                "dead": rb_gn_dead, "neg_gamma": rb_neg_gamma,
                "gamma_mean": float(rb_bn_gamma.abs().mean().item()),
                "gamma_std": float(rb_bn_gamma.std().item()),
            }
            rb_conv2_w = net.res_blocks[bi].conv2.weight.data.flatten(1)
            rb_svs = torch.linalg.svdvals(rb_conv2_w)
            rb_energy = (rb_svs**2).cumsum(0) / (rb_svs**2).sum()
            rb_rank90 = int((rb_energy <= 0.9).sum().item()) + 1
            all_rb_bn[bi]["svd_rank90"] = rb_rank90
            all_rb_bn[bi]["svd_total"] = rb_conv2_w.shape[0]

        # final_bn (GroupNorm) gamma tracking
        fbn_dead = -1
        fbn_gamma_mean = fbn_gamma_std = 0.0
        try:
            fbn = getattr(net, 'final_bn', None)
            if fbn is not None:
                fbn_gamma = fbn.weight.data
                fbn_gamma_mean = float(fbn_gamma.abs().mean().item())
                fbn_gamma_std = float(fbn_gamma.std().item())
                fbn_dead = int((fbn_gamma.abs() < 0.1).sum().item())
            else:
                print("  [FBN-DBG] final_bn not found on net")
        except Exception as e:
            print(f"  [FBN-DBG] Exception in FBN diagnostic: {e}")

        vh_diag.update({
            "svd_bb_rank90": bb_rank90, "svd_bb_rank99": bb_rank99,
            "svd_bb_total": bb_n, "svd_bb_near_zero": bb_near_zero_sv,
            "gn_dead_deepest": gn_dead,
            "svd_pfc_rank90": pfc_rank90, "svd_pfc_rank99": pfc_rank99,
            "svd_pfc_max_rank": pfc_max_rank,
            "pconv_ch_norms": pc_ch_norms, "all_rb_bn": all_rb_bn,
            "final_bn_dead": fbn_dead,
            "final_bn_gamma_mean": fbn_gamma_mean,
            "final_bn_gamma_std": fbn_gamma_std,
        })
        net.train()
    except (RuntimeError, ValueError, torch.linalg.LinAlgError) as e:
        print(f"  [DIAG-DBG] SVD/rank diagnostic block failed: {e}")


def compute_gradient_conflict_diagnostic(trainer):
    """Measure gradient conflict between x_wins_next and o_wins_next FixedEval positions.

    Two separate forward+backward passes, then cosine similarity of parameter gradients.
    Does NOT update weights. Returns dict with cosine metrics, or empty dict on failure.
    """
    net = trainer.net
    device = trainer.device
    result = {}

    fe_inputs = getattr(trainer, '_fixed_eval_inputs', None)
    fe_names = getattr(trainer, '_fixed_eval_names', None)
    if fe_inputs is None or fe_names is None:
        return result

    try:
        idx_x = fe_names.index('x_wins_next')
        idx_o = fe_names.index('o_wins_next')
    except ValueError:
        return result

    try:
        net.eval()
        win_target = torch.LongTensor([0]).to(device)  # WDL class 0 = win

        # Forward + backward for x_wins_next
        trainer.optimizer.zero_grad()
        x_pred, _ = net(fe_inputs[idx_x:idx_x + 1])
        x_vloss = F.cross_entropy(x_pred, win_target)
        x_vloss.backward()
        x_grads = []
        for p in net.parameters():
            if p.grad is not None:
                x_grads.append(p.grad.detach().clone().flatten())
            else:
                x_grads.append(torch.zeros(p.numel(), device=device))
        x_grad_vec = torch.cat(x_grads)

        # Forward + backward for o_wins_next
        trainer.optimizer.zero_grad()
        o_pred, _ = net(fe_inputs[idx_o:idx_o + 1])
        o_vloss = F.cross_entropy(o_pred, win_target)
        o_vloss.backward()
        o_grads = []
        for p in net.parameters():
            if p.grad is not None:
                o_grads.append(p.grad.detach().clone().flatten())
            else:
                o_grads.append(torch.zeros(p.numel(), device=device))
        o_grad_vec = torch.cat(o_grads)

        # Clean up
        trainer.optimizer.zero_grad()
        net.train()

        # Full-parameter cosine similarity
        cos_all = float(F.cosine_similarity(
            x_grad_vec.unsqueeze(0), o_grad_vec.unsqueeze(0)).item())

        # Per-component: backbone vs value head
        backbone_x, backbone_o = [], []
        value_x, value_o = [], []
        gi = 0
        for name, p in net.named_parameters():
            n = p.numel()
            xg = x_grad_vec[gi:gi + n]
            og = o_grad_vec[gi:gi + n]
            if name.startswith('value'):
                value_x.append(xg)
                value_o.append(og)
            elif not name.startswith('policy'):
                backbone_x.append(xg)
                backbone_o.append(og)
            gi += n

        cos_backbone = 0.0
        if backbone_x:
            bx = torch.cat(backbone_x).unsqueeze(0)
            bo = torch.cat(backbone_o).unsqueeze(0)
            cos_backbone = float(F.cosine_similarity(bx, bo).item())

        cos_value = 0.0
        if value_x:
            vx = torch.cat(value_x).unsqueeze(0)
            vo = torch.cat(value_o).unsqueeze(0)
            cos_value = float(F.cosine_similarity(vx, vo).item())

        result = {
            'cos_all': cos_all,
            'cos_backbone': cos_backbone,
            'cos_value_head': cos_value,
            'x_pred_scalar': float(wdl_to_scalar(x_pred.detach()).item()),
            'o_pred_scalar': float(wdl_to_scalar(o_pred.detach()).item()),
        }
    except (RuntimeError, ValueError, IndexError) as e:
        print(f"  [DIAG-DBG] Gradient conflict diagnostic failed: {e}")
        trainer.optimizer.zero_grad()
        net.train()

    return result


def detect_immediate_wins(states):
    """Detect which batch positions have an immediate winning move for current player.

    Operates on the training batch tensor [B, C, H, W] (ch0=my pieces, ch1=opp pieces).
    For each column, simulates placing a piece and checks for 4-in-a-row.

    Returns:
        torch.BoolTensor of shape [B], True if position has an immediate win.
    """
    B, C, H, W = states.shape
    my_pieces = states[:, 0]       # [B, H, W]
    opp_pieces = states[:, 1]      # [B, H, W]
    occupied = my_pieces + opp_pieces

    has_imm_win = torch.zeros(B, dtype=torch.bool, device=states.device)

    for col in range(W):
        col_occupied = occupied[:, :, col]          # [B, H]
        n_occupied = col_occupied.sum(dim=1).long()  # [B]
        valid = (n_occupied < H)
        if not valid.any():
            continue

        # Simulate placing current player's piece at lowest free row
        sim_my = my_pieces.clone()
        rows = n_occupied.clamp(max=H - 1)
        batch_idx = torch.arange(B, device=states.device)
        sim_my[batch_idx[valid], rows[valid], col] = 1.0

        # Check 4-in-a-row: horizontal, vertical, both diagonals
        win = torch.zeros(B, dtype=torch.bool, device=states.device)
        if W >= 4:  # horizontal
            h = sim_my[:, :, :-3] * sim_my[:, :, 1:-2] * sim_my[:, :, 2:-1] * sim_my[:, :, 3:]
            win = win | h.any(dim=2).any(dim=1)
        if H >= 4:  # vertical
            v = sim_my[:, :-3, :] * sim_my[:, 1:-2, :] * sim_my[:, 2:-1, :] * sim_my[:, 3:, :]
            win = win | v.any(dim=2).any(dim=1)
        if H >= 4 and W >= 4:  # diagonal ↘
            d1 = sim_my[:, :-3, :-3] * sim_my[:, 1:-2, 1:-2] * sim_my[:, 2:-1, 2:-1] * sim_my[:, 3:, 3:]
            win = win | d1.any(dim=2).any(dim=1)
        if H >= 4 and W >= 4:  # diagonal ↗
            d2 = sim_my[:, 3:, :-3] * sim_my[:, 2:-1, 1:-2] * sim_my[:, 1:-2, 2:-1] * sim_my[:, :-3, 3:]
            win = win | d2.any(dim=2).any(dim=1)

        has_imm_win = has_imm_win | (win & valid)

    return has_imm_win
