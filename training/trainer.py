import math
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from training.replay_buffer import ReplayBuffer
from training.parallel_self_play import BatchedSelfPlay
from training.diagnostics import (
    raw_value_to_wdl_class, compute_three_in_a_row, compute_buffer_diagnostics,
    compute_pre_training_diagnostics, compute_post_training_player_diagnostics,
    compute_value_head_diagnostics, compute_backbone_gradient_decomposition,
    compute_svd_rank_diagnostics,
)


class Trainer:
    def __init__(self, game, net, config=None):
        self.game = game
        self.net = net
        self.config = config or {}
        self.num_simulations = self.config.get("num_simulations", 50)
        self.games_per_iteration = self.config.get("games_per_iteration", 2)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        self.batch_size = self.config.get("batch_size", 64)
        self.epochs = self.config.get("epochs", 10)
        self.lr = self.config.get("lr", 0.01)
        self.device = self.config.get("device", "cpu")
        self.max_train_steps = self.config.get("max_train_steps", 5000)
        self.target_epochs = self.config.get("target_epochs", 4)
        self.train_ratio = self.config.get("train_ratio", 0)  # gradient steps per new position; 0 = use epoch-based
        self.global_step = 0  # global training step counter (persists across iterations)
        self.buffer = ReplayBuffer(self.config.get("buffer_size", 100000))

        # Separate params: apply weight decay only to conv/linear weights,
        # NOT to BatchNorm gamma/beta or bias terms.  Decaying BN gamma
        # shrinks activations every step (compounding through layers) while
        # BN just rescales — it doesn't regularize, it just causes magnitude decay.
        decay_params = []
        no_decay_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if 'bn' in name or 'bias' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        self.optimizer = torch.optim.SGD([
            {'params': decay_params, 'weight_decay': 5e-4},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=self.lr, momentum=0.9)

        self.value_loss_weight = self.config.get("value_loss_weight", 1.0)

        # Mixed precision training: FP16 forward/loss, FP32 gradients
        self.use_amp = (self.device == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        game_name = self.config.get("game_name", "unknown")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = self.config.get("log_dir", f"runs/{game_name}/{timestr}")
        self.writer = SummaryWriter(log_dir)

    def train_network(self, n_new_positions=0):
        """Train the network on samples from the replay buffer."""
        samples = [s for s in self.buffer.arr if s is not None]
        if len(samples) < self.batch_size:
            print(f"  Not enough samples ({len(samples)}), skipping training")
            return None

        # Buffer diagnostics
        all_values = np.array([s[2] for s in samples])
        buf_diag = compute_buffer_diagnostics(samples, all_values)

        # Setup: split, pools, accumulators, schedule, pre-training diagnostics
        setup = self._init_training_state(samples, n_new_positions)
        samples = setup['train_samples']
        acc = setup['acc']
        cfg = {k: setup[k] for k in ('num_steps', 'effective_vlw', 'effective_epochs',
                                       'early_cutoff', 'late_start', 'lr_min')}

        self.net.train()
        for step in range(cfg['num_steps']):
            # Global cosine annealing: lr decays over entire training run
            total = max(self.global_total_steps, 1)
            lr = cfg['lr_min'] + 0.5 * (self.lr - cfg['lr_min']) * (
                1 + math.cos(math.pi * self.global_step / total))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.global_step += 1

            # Sample batch (stratified or random)
            if (setup['use_stratified'] and len(setup['x_pool']) >= setup['half_batch']
                    and len(setup['o_pool']) >= setup['half_batch']):
                batch = (random.sample(setup['x_pool'], k=setup['half_batch'])
                         + random.sample(setup['o_pool'], k=setup['half_batch']))
            else:
                batch = random.sample(samples, k=min(self.batch_size, len(samples)))

            # Data prep
            t0 = time.time()
            states = torch.FloatTensor(np.array([s[0] for s in batch])).to(self.device)
            target_pis = torch.FloatTensor(np.array([s[1] for s in batch])).to(self.device)
            raw_v = np.array([s[2] for s in batch])
            target_vs = torch.LongTensor(raw_value_to_wdl_class(raw_v)).to(self.device)
            acc['data_prep_time'] += time.time() - t0

            # Forward + backward
            t0 = time.time()
            with torch.autocast('cuda', enabled=self.use_amp):
                pred_vs, pred_pi_logits = self.net(states)
                value_loss = F.cross_entropy(pred_vs, target_vs)
                log_pred_pis = F.log_softmax(pred_pi_logits, dim=1)
                policy_loss = -torch.mean(torch.sum(target_pis * log_pred_pis, dim=1))
                loss = cfg['effective_vlw'] * value_loss + policy_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Collect diagnostics (reads unscaled gradients)
            self._collect_step_diagnostics(
                step, states, target_vs, target_pis,
                pred_vs, pred_pi_logits, value_loss, policy_loss, acc, cfg)

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            acc['gradient_time'] += time.time() - t0

            # Accumulate losses
            acc['total_loss'] += loss.item()
            acc['total_value_loss'] += value_loss.item()
            acc['total_policy_loss'] += policy_loss.item()
            acc['num_batches'] += 1

        # Aggregate results + post-training diagnostics
        return self._aggregate_training_results(
            acc, samples, setup['val_samples'], buf_diag, cfg, setup)

    def _init_training_state(self, samples, n_new_positions):
        """Prepare train/val split, tracking accumulators, and LR schedule params."""
        # Split into train/val for overfitting detection (90/10)
        random.shuffle(samples)
        val_size = max(len(samples) // 10, self.batch_size)
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]

        # Stratified sampling pools: split by player to ensure 50/50 X/O batches
        x_pool = [s for s in train_samples if s[0][2, 0, 0] < 0]
        o_pool = [s for s in train_samples if s[0][2, 0, 0] > 0]

        # Dynamic training steps
        n_samples = len(train_samples)
        buffer_capacity = self.buffer.max_size
        fill_ratio = min(n_samples / max(buffer_capacity, 1), 1.0)
        if self.train_ratio > 0 and n_new_positions > 0:
            target_steps = int(n_new_positions * self.train_ratio / self.batch_size)
            num_steps = max(1, target_steps)
        else:
            scaled_epochs = 1.0 + (self.target_epochs - 1.0) * fill_ratio
            target_steps = int(scaled_epochs * (n_samples // self.batch_size))
            num_steps = max(1, min(self.max_train_steps, target_steps))

        effective_vlw = 1.0 + (self.value_loss_weight - 1.0) * fill_ratio
        effective_epochs = (num_steps * self.batch_size) / n_samples
        early_cutoff = max(num_steps // 10, 1)
        late_start = num_steps - early_cutoff
        lr_min = self.lr * 0.1

        # Pre-training value loss + per-player predictions
        pre_train_vloss, pbias_data = compute_pre_training_diagnostics(
            self.net, train_samples, self.device
        )

        # All tracking accumulators
        acc = {
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
            'grad_correct_count': 0, 'grad_total_count': 0,
            'all_policy_entropy': [],
            'top1_correct_sum': 0, 'top3_correct_sum': 0, 'policy_acc_count': 0,
            'confident_correct_sum': 0, 'confident_total': 0,
            'all_rb_grad_norms': {},
            'sub_iter_log': [],
            'conf_buckets': {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0},
            'conf_total': 0,
            'fixed_eval_trajectory': [],
            'phase_vloss_sums': {'early': 0.0, 'mid': 0.0, 'late': 0.0},
            'phase_counts': {'early': 0, 'mid': 0, 'late': 0},
            'ploss_decisive_sum': 0.0, 'ploss_ambiguous_sum': 0.0,
            'decisive_count': 0, 'ambiguous_count': 0,
        }

        return {
            'train_samples': train_samples, 'val_samples': val_samples,
            'x_pool': x_pool, 'o_pool': o_pool,
            'use_stratified': len(x_pool) > 0 and len(o_pool) > 0,
            'half_batch': self.batch_size // 2,
            'num_steps': num_steps, 'effective_vlw': effective_vlw,
            'effective_epochs': effective_epochs,
            'early_cutoff': early_cutoff, 'late_start': late_start,
            'lr_min': lr_min, 'n_samples': n_samples, 'fill_ratio': fill_ratio,
            'pre_train_vloss': pre_train_vloss, 'pbias_data': pbias_data,
            'acc': acc,
        }

    def _collect_step_diagnostics(self, step, states, target_vs, target_pis,
                                   pred_vs, pred_pi_logits, value_loss, policy_loss,
                                   acc, cfg):
        """Sample diagnostics at various intervals during training. Mutates acc."""
        num_steps = cfg['num_steps']
        early_cutoff = cfg['early_cutoff']
        late_start = cfg['late_start']

        # (A) Per-player loss breakdown every 10 steps
        if step % 10 == 0:
            with torch.no_grad():
                my_counts = states[:, 0].sum(dim=(1, 2))
                opp_counts = states[:, 1].sum(dim=(1, 2))
                is_x = (my_counts == opp_counts)
                is_o = ~is_x

                per_sample_vloss = F.cross_entropy(pred_vs, target_vs, reduction='none')
                wdl_probs = F.softmax(pred_vs, dim=1)
                scalar_v = wdl_probs[:, 0] - wdl_probs[:, 2]
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

                # (#6) Value loss by game phase
                total_pieces = my_counts + opp_counts
                for phase, lo, hi in [('early', 0, 8), ('mid', 9, 20), ('late', 21, 999)]:
                    mask = (total_pieces >= lo) & (total_pieces <= hi)
                    if mask.any():
                        acc['phase_vloss_sums'][phase] += per_sample_vloss[mask].mean().item()
                        acc['phase_counts'][phase] += 1

        # (F) Gradient direction check every 50 steps
        if step % 50 == 0:
            with torch.no_grad():
                acc['grad_correct_count'] += 1
                acc['grad_total_count'] += 1
                wdl_p = F.softmax(pred_vs, dim=1)
                sv = wdl_p[:, 0] - wdl_p[:, 2]
                st = 1 - target_vs.float()
                error_scalar = sv - st
                fc1_grad = self.net.value_fc1.weight.grad
                fc2_grad = self.net.value_fc2.weight.grad
                if fc1_grad is not None and fc2_grad is not None:
                    if not hasattr(self, '_grad_stats'):
                        self._grad_stats = []
                    fc1_per_neuron_gnorm = fc1_grad.norm(dim=1).cpu().numpy()
                    vconv_g = self.net.value_conv.weight.grad
                    vbn_g = self.net.value_bn.weight.grad
                    self._grad_stats.append({
                        'fc1_grad_mean': fc1_grad.mean().item(),
                        'fc1_grad_std': fc1_grad.std().item(),
                        'fc1_grad_norm': fc1_grad.norm().item(),
                        'fc2_grad_mean': fc2_grad.mean().item(),
                        'fc2_grad_std': fc2_grad.std().item(),
                        'fc2_grad_norm': fc2_grad.norm().item(),
                        'pred_mean': sv.mean().item(),
                        'target_mean': st.mean().item(),
                        'error_mean': error_scalar.mean().item(),
                        'fc1_per_neuron_gnorm': fc1_per_neuron_gnorm,
                        'vconv_grad_norm': vconv_g.norm().item() if vconv_g is not None else 0.0,
                        'vbn_gamma_grad_norm': vbn_g.norm().item() if vbn_g is not None else 0.0,
                    })

        # Sample gradient norms every 100 steps
        if step % 100 == 0:
            v_norm = 0.0
            p_norm = 0.0
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    g = param.grad.norm().item()
                    if "value" in name:
                        v_norm += g ** 2
                    elif "policy" in name:
                        p_norm += g ** 2
            acc['value_grad_norms'].append(v_norm ** 0.5)
            acc['policy_grad_norms'].append(p_norm ** 0.5)

            # (RB) Per-residual-block gradient norms
            for i, block in enumerate(self.net.res_blocks):
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
            with torch.no_grad():
                _wdl_sub = F.softmax(pred_vs.detach(), dim=1)
                _sv_sub = (_wdl_sub[:, 0] - _wdl_sub[:, 2])
                _mean_conf = _sv_sub.abs().mean().item()
                _mean_v = _sv_sub.mean().item()
            acc['sub_iter_log'].append({
                'step': step,
                'vloss': value_loss.item(),
                'ploss': policy_loss.item(),
                'v_grad': acc['value_grad_norms'][-1] if acc['value_grad_norms'] else 0,
                'p_grad': acc['policy_grad_norms'][-1] if acc['policy_grad_norms'] else 0,
                'mean_conf': _mean_conf,
                'mean_v': _mean_v,
            })

        # Intra-iteration value trajectory on FixedEval positions
        _fe_inputs = getattr(self, '_fixed_eval_inputs', None)
        _fe_names = getattr(self, '_fixed_eval_names', None)
        if _fe_inputs is not None and _fe_names and (step % 300 == 0 or step == num_steps - 1):
            self.net.eval()
            with torch.no_grad():
                _fe_v, _fe_p = self.net(_fe_inputs)
                _fe_probs = F.softmax(_fe_v, dim=1)
                _fe_vals = (_fe_probs[:, 0] - _fe_probs[:, 2]).cpu().numpy()
            self.net.train()
            _fe_entry = {'step': step}
            for _fi, _fn in enumerate(_fe_names):
                _fe_entry[_fn] = float(_fe_vals[_fi])
            acc['fixed_eval_trajectory'].append(_fe_entry)

        # Track early vs late loss
        if step < early_cutoff:
            acc['early_vloss'] += value_loss.item()
            acc['early_ploss'] += policy_loss.item()
        if step >= late_start:
            acc['late_vloss'] += value_loss.item()
            acc['late_ploss'] += policy_loss.item()

        # Sample predictions from last 10% of steps for distribution analysis
        if step >= late_start:
            with torch.no_grad():
                wdl_p_late = F.softmax(pred_vs.detach(), dim=1)
                scalar_v_late = (wdl_p_late[:, 0] - wdl_p_late[:, 2]).cpu().numpy()
            acc['all_pred_vs'].append(scalar_v_late)

            # Value confidence distribution buckets
            _abs_v = np.abs(scalar_v_late)
            acc['conf_total'] += len(_abs_v)
            acc['conf_buckets']['very_low'] += int((_abs_v < 0.1).sum())
            acc['conf_buckets']['low'] += int(((_abs_v >= 0.1) & (_abs_v < 0.3)).sum())
            acc['conf_buckets']['medium'] += int(((_abs_v >= 0.3) & (_abs_v < 0.6)).sum())
            acc['conf_buckets']['high'] += int(((_abs_v >= 0.6) & (_abs_v < 0.9)).sum())
            acc['conf_buckets']['very_high'] += int((_abs_v >= 0.9).sum())

            # (P) Policy quality metrics
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

                # (C) Value confidence calibration
                sv_conf = wdl_p_late[:, 0] - wdl_p_late[:, 2]
                scalar_tgt = 1 - target_vs.float()
                confident_mask = sv_conf.abs() > 0.5
                if confident_mask.any():
                    confident_signs_correct = (
                        sv_conf[confident_mask].sign() == scalar_tgt[confident_mask].sign()
                    ).float()
                    acc['confident_correct_sum'] += confident_signs_correct.sum().item()
                    acc['confident_total'] += confident_mask.sum().item()

                # (#8) Policy loss on decisive vs ambiguous positions
                decisive_mask = (target_vs != 1)
                ambig_mask = (target_vs == 1)
                per_sample_ploss = -torch.sum(target_pis * log_pi, dim=1)
                if decisive_mask.any():
                    acc['ploss_decisive_sum'] += per_sample_ploss[decisive_mask].mean().item()
                    acc['decisive_count'] += 1
                if ambig_mask.any():
                    acc['ploss_ambiguous_sum'] += per_sample_ploss[ambig_mask].mean().item()
                    acc['ambiguous_count'] += 1

    def _aggregate_training_results(self, acc, samples, val_samples, buf_diag, cfg, setup):
        """Compute averages, run validation, value head diagnostics. Returns (loss, vloss, ploss)."""
        num_batches = acc['num_batches']
        early_cutoff = cfg['early_cutoff']

        avg_loss = acc['total_loss'] / max(num_batches, 1)
        avg_value_loss = acc['total_value_loss'] / max(num_batches, 1)
        avg_policy_loss = acc['total_policy_loss'] / max(num_batches, 1)
        early_vloss = acc['early_vloss'] / max(early_cutoff, 1)
        early_ploss = acc['early_ploss'] / max(early_cutoff, 1)
        late_vloss = acc['late_vloss'] / max(early_cutoff, 1)
        late_ploss = acc['late_ploss'] / max(early_cutoff, 1)

        # Value prediction distribution (from last 10% of training)
        all_pred_vs = acc['all_pred_vs']
        pred_v_all = np.concatenate(all_pred_vs) if all_pred_vs else np.array([0.0])
        pred_v_mean = pred_v_all.mean()
        pred_v_std = pred_v_all.std()
        pred_v_abs_mean = np.abs(pred_v_all).mean()

        # Policy quality metrics
        avg_policy_entropy = float(np.mean(acc['all_policy_entropy'])) if acc['all_policy_entropy'] else 0.0
        policy_top1_acc = acc['top1_correct_sum'] / max(acc['policy_acc_count'], 1)
        policy_top3_acc = acc['top3_correct_sum'] / max(acc['policy_acc_count'], 1)

        # Value confidence calibration
        value_confidence_acc = acc['confident_correct_sum'] / max(acc['confident_total'], 1)
        value_confident_frac = acc['confident_total'] / max(acc['policy_acc_count'], 1)

        # Per-block gradient norms
        avg_rb_grad_norms = {}
        for i, norms in acc['all_rb_grad_norms'].items():
            avg_rb_grad_norms[i] = float(np.mean(norms))

        # Held-out validation loss
        val_vloss = 0.0
        val_ploss = 0.0
        self.net.eval()
        with torch.no_grad():
            val_batches = 0
            for i in range(0, len(val_samples), self.batch_size):
                vb = val_samples[i:i + self.batch_size]
                if len(vb) < 2:
                    continue
                vs = torch.FloatTensor(np.array([s[0] for s in vb])).to(self.device)
                vt_pi = torch.FloatTensor(np.array([s[1] for s in vb])).to(self.device)
                vt_raw = np.array([s[2] for s in vb])
                vt_v = torch.LongTensor(raw_value_to_wdl_class(vt_raw)).to(self.device)
                pv, pp_logits = self.net(vs)
                val_vloss += F.cross_entropy(pv, vt_v).item()
                val_ploss += -torch.mean(torch.sum(vt_pi * F.log_softmax(pp_logits, dim=1), dim=1)).item()
                val_batches += 1
            if val_batches > 0:
                val_vloss /= val_batches
                val_ploss /= val_batches
        self.net.train()

        policy_frac = avg_policy_loss / max(avg_loss, 1e-8)
        avg_value_grad = np.mean(acc['value_grad_norms']) if acc['value_grad_norms'] else 0.0
        avg_policy_grad = np.mean(acc['policy_grad_norms']) if acc['policy_grad_norms'] else 0.0

        buffer_fill = sum(1 for s in self.buffer.arr if s is not None)
        buffer_full = buffer_fill >= self.buffer.max_size

        # Theoretical value loss floor
        val_loss_floor = 0.0
        for frac in [buf_diag['frac_neg'], buf_diag['frac_draw'], buf_diag['frac_pos']]:
            if frac > 0:
                val_loss_floor -= frac * np.log(frac)

        # Per-player averages
        x_vloss_avg = acc['x_vloss_sum'] / max(acc['x_count'], 1)
        o_vloss_avg = acc['o_vloss_sum'] / max(acc['o_count'], 1)
        x_target_avg = acc['x_target_sum'] / max(acc['x_count'], 1)
        o_target_avg = acc['o_target_sum'] / max(acc['o_count'], 1)
        x_pred_avg = acc['x_pred_sum'] / max(acc['x_count'], 1)
        o_pred_avg = acc['o_pred_sum'] / max(acc['o_count'], 1)

        # Gradient stats summary
        grad_stats_summary = {}
        if hasattr(self, '_grad_stats') and self._grad_stats:
            gs = self._grad_stats
            all_per_neuron = np.stack([g['fc1_per_neuron_gnorm'] for g in gs])
            avg_per_neuron_gnorm = all_per_neuron.mean(axis=0)
            grad_stats_summary = {
                'fc1_grad_norm_mean': np.mean([g['fc1_grad_norm'] for g in gs]),
                'fc2_grad_norm_mean': np.mean([g['fc2_grad_norm'] for g in gs]),
                'fc1_grad_mean': np.mean([g['fc1_grad_mean'] for g in gs]),
                'fc2_grad_mean': np.mean([g['fc2_grad_mean'] for g in gs]),
                'error_mean_trend': [g['error_mean'] for g in gs],
                'fc1_per_neuron_gnorm': avg_per_neuron_gnorm,
                'vconv_grad_norm': np.mean([g['vconv_grad_norm'] for g in gs]),
                'vbn_gamma_grad_norm': np.mean([g['vbn_gamma_grad_norm'] for g in gs]),
            }
            self._grad_stats = []

        # Value head health diagnostics
        vh_diag = compute_value_head_diagnostics(self, samples, grad_stats_summary)
        compute_backbone_gradient_decomposition(self, samples, vh_diag)
        compute_svd_rank_diagnostics(self.net, vh_diag)

        # Post-training per-player diagnostics
        pbias_data = setup['pbias_data']
        _post_x_pred, _post_o_pred, _post_x_acc, _post_o_acc = \
            compute_post_training_player_diagnostics(self.net, pbias_data, self.device)

        pre_train_vloss = setup['pre_train_vloss']
        phase_vloss_sums = acc['phase_vloss_sums']
        phase_counts = acc['phase_counts']

        self._train_perf = {
            "data_prep_time": acc['data_prep_time'],
            "gradient_time": acc['gradient_time'],
            "num_samples": setup['n_samples'],
            "num_batches": num_batches,
        }
        self._train_diag = {
            "val_target_mean": buf_diag['val_mean'],
            "val_target_std": buf_diag['val_std'],
            "frac_pos": buf_diag['frac_pos'],
            "frac_neg": buf_diag['frac_neg'],
            "frac_draw": buf_diag['frac_draw'],
            "effective_epochs": cfg['effective_epochs'],
            "num_steps": cfg['num_steps'],
            "early_vloss": early_vloss, "early_ploss": early_ploss,
            "late_vloss": late_vloss, "late_ploss": late_ploss,
            "val_vloss": val_vloss, "val_ploss": val_ploss,
            "buffer_fill": buffer_fill, "buffer_capacity": self.buffer.max_size,
            "buffer_full": buffer_full,
            "pred_v_mean": pred_v_mean, "pred_v_std": pred_v_std,
            "pred_v_abs_mean": pred_v_abs_mean,
            "policy_grad_frac": policy_frac,
            "val_loss_floor": val_loss_floor,
            "avg_value_grad_norm": avg_value_grad,
            "avg_policy_grad_norm": avg_policy_grad,
            "x_vloss": x_vloss_avg, "o_vloss": o_vloss_avg,
            "x_target_mean": x_target_avg, "o_target_mean": o_target_avg,
            "x_pred_mean": x_pred_avg, "o_pred_mean": o_pred_avg,
            "grad_stats": grad_stats_summary,
            "vh_diag": vh_diag,
            "effective_vlw": cfg['effective_vlw'],
            "policy_entropy": avg_policy_entropy,
            "policy_top1_acc": policy_top1_acc, "policy_top3_acc": policy_top3_acc,
            "value_confidence_acc": value_confidence_acc,
            "value_confident_frac": value_confident_frac,
            "rb_grad_norms": avg_rb_grad_norms,
            "val_hist": buf_diag['val_hist'],
            "three_r_diag": buf_diag['three_r_diag'],
            "pre_train_vloss": pre_train_vloss,
            "vloss_delta": (val_vloss - pre_train_vloss) if pre_train_vloss is not None else None,
            "phase_vloss_early": phase_vloss_sums['early'] / max(phase_counts['early'], 1),
            "phase_vloss_mid": phase_vloss_sums['mid'] / max(phase_counts['mid'], 1),
            "phase_vloss_late": phase_vloss_sums['late'] / max(phase_counts['late'], 1),
            "phase_counts": phase_counts,
            "policy_loss_decisive": acc['ploss_decisive_sum'] / max(acc['decisive_count'], 1),
            "policy_loss_ambiguous": acc['ploss_ambiguous_sum'] / max(acc['ambiguous_count'], 1),
            "decisive_frac": acc['decisive_count'] / max(acc['decisive_count'] + acc['ambiguous_count'], 1),
            "sub_iter_log": acc['sub_iter_log'],
            "conf_dist": {k: v / max(acc['conf_total'], 1) for k, v in acc['conf_buckets'].items()},
            "fixed_eval_trajectory": acc['fixed_eval_trajectory'],
            "buf_n_x": buf_diag['n_x_buf'], "buf_n_o": buf_diag['n_o_buf'],
            "buf_mean_tgt_x": buf_diag['mean_tgt_x'], "buf_mean_tgt_o": buf_diag['mean_tgt_o'],
            "buf_frac_pos_x": buf_diag['frac_pos_x'], "buf_frac_pos_o": buf_diag['frac_pos_o'],
            "pbias_pre_x_pred": pbias_data['pre_x_pred'] if pbias_data else 0.0,
            "pbias_pre_o_pred": pbias_data['pre_o_pred'] if pbias_data else 0.0,
            "pbias_pre_x_acc": pbias_data['pre_x_acc'] if pbias_data else 0.0,
            "pbias_pre_o_acc": pbias_data['pre_o_acc'] if pbias_data else 0.0,
            "pbias_post_x_pred": _post_x_pred, "pbias_post_o_pred": _post_o_pred,
            "pbias_post_x_acc": _post_x_acc, "pbias_post_o_acc": _post_o_acc,
        }
        return avg_loss, avg_value_loss, avg_policy_loss

    def _self_play(self, iteration):
        """Run self-play games in parallel with batched evaluation."""
        self._batched = BatchedSelfPlay(
            self.game, self.net, self.games_per_iteration, self.num_simulations,
            selects_per_round=self.config.get("selects_per_round", 1),
            vl_value=self.config.get("vl_value", 0.0),
            temp_threshold=self.config.get("temp_threshold", 15),
            c_puct=self.config.get("c_puct", 1.5),
            dirichlet_alpha=self.config.get("dirichlet_alpha", 1.0),
            tree_reuse=self.config.get("tree_reuse", True),
            resign_threshold=self.config.get("resign_threshold", -1.0),
            resign_min_moves=self.config.get("resign_min_moves", 99),
            resign_check_prob=self.config.get("resign_check_prob", 0.0),
        )
        return self._batched.play_games()

    def run(self, num_iterations=1):
        """Run the training loop: self-play -> train -> save."""
        # Estimate total training steps for global LR schedule
        # (rough: assumes ~200 steps/iter at full buffer with ratio-based training)
        avg_game_len = 25  # approximate average game length for Connect4
        n_syms = 2  # mirror augmentation factor
        est_new_per_iter = self.games_per_iteration * avg_game_len * n_syms
        if self.train_ratio > 0:
            est_steps_per_iter = max(1, int(est_new_per_iter * self.train_ratio / self.batch_size))
        else:
            est_steps_per_iter = min(self.max_train_steps,
                                     int(self.target_epochs * self.buffer.max_size / self.batch_size))
        self.global_total_steps = est_steps_per_iter * num_iterations
        for iteration in range(num_iterations):
            iter_t0 = time.time()

            # Self-play
            t0 = time.time()
            all_examples, results, game_lengths = self._self_play(iteration)
            self_play_time = time.time() - t0

            # Augment with symmetries (e.g. left-right mirror for Connect4)
            augmented = []
            for ex in all_examples:
                for sym_input, sym_policy in self.game.get_symmetries(ex[0], ex[1]):
                    augmented.append([sym_input, sym_policy, ex[2]])
            n_new_positions = len(augmented)
            self.buffer.insert_batch(augmented)

            # === Per-iteration 3-in-a-row target bias (fresh batch only) ===
            iter_3r = None
            iter_3r_sign = None
            try:
                _iter_inputs = np.array([ex[0] for ex in augmented])
                _iter_targets = np.array([ex[2] for ex in augmented])
                _3r = compute_three_in_a_row(_iter_inputs, _iter_targets, sign_split=True)
                iter_3r = {
                    'mine': {'count': _3r['mine']['count'], 'mean': _3r['mine']['mean_target']},
                    'opp': {'count': _3r['opp']['count'], 'mean': _3r['opp']['mean_target']},
                }
                iter_3r_sign = _3r.get('sign_split')
            except (IndexError, ValueError, RuntimeError) as e:
                print(f"  [DIAG-DBG] Per-iteration 3-in-a-row failed: {e}")

            # Log self-play stats
            wins_p1 = results.count(-1)
            wins_p2 = results.count(1)
            draws = results.count(0)
            avg_length = np.mean(game_lengths)
            min_length = int(np.min(game_lengths))
            max_length = int(np.max(game_lengths))
            p1_win_pct = wins_p1 / max(len(results), 1)
            # === Pre-training diagnostics ===
            self._eval_diagnostic_positions(iteration, prefix="pre_", label="PreTrainEval")

            # Pre-training segregation (weight-based, no forward pass needed)
            pre_seg = None
            try:
                _val_w = self.net.value_conv.weight.data
                _pol_w = self.net.policy_conv.weight.data
                _val_ch = _val_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()
                _pol_ch = _pol_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()
                pre_vp_corr = float(np.corrcoef(_val_ch, _pol_ch)[0, 1])
                _pre_top20_v = np.argsort(_val_ch)[-20:]
                _pre_top20_p = np.argsort(_pol_ch)[-20:]
                pre_overlap = len(set(_pre_top20_v.tolist()) & set(_pre_top20_p.tolist()))
                pre_seg = {'vp_corr': pre_vp_corr, 'overlap': pre_overlap}
            except (RuntimeError, ValueError, IndexError) as e:
                print(f"  [DIAG-DBG] Pre-training segregation failed: {e}")

            # === Snapshot backbone features + weights BEFORE training ===
            # (1) Backbone feature drift: run FixedEval positions through backbone
            # (3) Per-block weight delta: snapshot conv2 weights
            _pre_bb_features = None
            _pre_block_weights = {}
            try:
                self.net.eval()
                _diag_positions = self._get_diagnostic_positions()
                if _diag_positions:
                    _diag_inputs = torch.FloatTensor(
                        np.array([p[1] for p in _diag_positions])
                    ).to(self.device)
                    with torch.no_grad():
                        _x = self.net.backbone_forward(_diag_inputs)
                        _pre_bb_features = _x.flatten(1).cpu()  # [5, channels*H*W]
                # Snapshot per-block conv2 weights
                for bi, block in enumerate(self.net.res_blocks):
                    _pre_block_weights[bi] = block.conv2.weight.data.clone()
                self.net.train()
            except Exception as e:
                print(f"  [DIAG-DBG] Pre-training snapshot failed: {e}")

            # (2) Store FixedEval inputs for intra-iteration trajectory
            self._fixed_eval_inputs = None
            self._fixed_eval_names = None
            try:
                if _diag_positions:
                    self._fixed_eval_inputs = torch.FloatTensor(
                        np.array([p[1] for p in _diag_positions])
                    ).to(self.device)
                    self._fixed_eval_names = [p[0] for p in _diag_positions]
            except (RuntimeError, ValueError, IndexError) as e:
                print(f"  [DIAG-DBG] Fixed eval inputs setup failed: {e}")

            # Train
            t0 = time.time()
            train_result = self.train_network(n_new_positions=n_new_positions)
            train_time = time.time() - t0

            # === Post-training diagnostics: backbone drift + weight delta ===
            drift_result = None
            try:
                if _pre_bb_features is not None and _diag_positions:
                    self.net.eval()
                    with torch.no_grad():
                        _post_bb_features = self.net.backbone_forward(_diag_inputs).flatten(1).cpu()
                    _cos_sims = F.cosine_similarity(_pre_bb_features, _post_bb_features, dim=1)
                    _pos_names = [p[0] for p in _diag_positions]
                    drift_result = {
                        'cos_mean': float(_cos_sims.mean()),
                        'cos_min': float(_cos_sims.min()),
                        'per_pos': {_pos_names[i]: float(_cos_sims[i]) for i in range(len(_cos_sims))},
                        'pos_names': _pos_names,
                    }
                    self.net.train()
            except Exception as e:
                print(f"  [DIAG-DBG] Backbone drift measurement failed: {e}")

            wdelta_result = None
            try:
                if _pre_block_weights:
                    wdelta_result = {}
                    for bi, block in enumerate(self.net.res_blocks):
                        _w_after = block.conv2.weight.data
                        _w_before = _pre_block_weights[bi]
                        _delta_norm = float((_w_after - _w_before).norm().item())
                        _before_norm = float(_w_before.norm().item())
                        wdelta_result[bi] = _delta_norm / max(_before_norm, 1e-10)
            except Exception as e:
                print(f"  [DIAG-DBG] Weight delta measurement failed: {e}")

            iter_time = time.time() - iter_t0

            # === Log all metrics ===
            self._log_iteration(iteration, num_iterations, {
                'train_result': train_result,
                'wins_p1': wins_p1, 'wins_p2': wins_p2, 'draws': draws,
                'avg_length': avg_length, 'min_length': min_length,
                'max_length': max_length, 'p1_win_pct': p1_win_pct,
                'self_play_time': self_play_time, 'train_time': train_time,
                'iter_time': iter_time,
                'iter_3r': iter_3r, 'iter_3r_sign': iter_3r_sign,
                'pre_seg': pre_seg, 'drift': drift_result,
                'wdelta': wdelta_result, 'n_new_positions': n_new_positions,
            })

            # Save every 5 iterations + always on the last one
            # Also save iteration 0 if no checkpoint exists (quick sanity check)
            no_checkpoint = not os.path.exists(os.path.join(self.checkpoint_dir, "latest.txt"))
            if (iteration + 1) % 5 == 0 or iteration == num_iterations - 1 or (iteration == 0 and no_checkpoint):
                self.net.save(self.checkpoint_dir, iteration=iteration, num_iterations=num_iterations)

        self.writer.close()

    def _log_iteration(self, iteration, num_iterations, stats):
        """Log all metrics for one iteration to console and TensorBoard."""
        writer = self.writer
        train_result = stats['train_result']

        self._log_self_play_stats(iteration, stats)

        # Main iteration summary
        if train_result is not None:
            avg_loss, avg_value_loss, avg_policy_loss = train_result
            writer.add_scalar("loss/total", avg_loss, iteration)
            writer.add_scalar("loss/value", avg_value_loss, iteration)
            writer.add_scalar("loss/policy", avg_policy_loss, iteration)
            print(f"  Iter {iteration+1}/{num_iterations}: loss={avg_loss:.4f} "
                  f"(v={avg_value_loss:.4f} p={avg_policy_loss:.4f}) | "
                  f"games: p1={stats['wins_p1']} p2={stats['wins_p2']} "
                  f"draw={stats['draws']} | "
                  f"avg_len={stats['avg_length']:.1f} "
                  f"({stats['min_length']}-{stats['max_length']}) | "
                  f"self_play={stats['self_play_time']:.1f}s "
                  f"train={stats['train_time']:.1f}s "
                  f"total={stats['iter_time']:.1f}s")

        writer.add_scalar("perf/self_play_time", stats['self_play_time'], iteration)
        writer.add_scalar("perf/train_time", stats['train_time'], iteration)

        self._log_mcts_perf(iteration)
        self._log_training_perf(iteration)
        self._log_training_diagnostics(iteration, stats)
        self._log_selfplay_value_diagnostics(iteration)
        self._log_intra_iteration_dynamics(iteration)
        self._eval_diagnostic_positions(iteration)

    def _log_self_play_stats(self, iteration, stats):
        """Log 3-in-a-row, self-play counts, pre-seg, drift, weight delta."""
        writer = self.writer

        iter_3r = stats.get('iter_3r')
        if iter_3r:
            _mine = iter_3r['mine']
            _opp = iter_3r['opp']
            print(f"  Diag[3R-iter]: fresh batch ch0: n={_mine['count']} target={_mine['mean']:+.3f} | "
                  f"ch1: n={_opp['count']} target={_opp['mean']:+.3f}")
            writer.add_scalar("diag/iter_3r_ch0_target", _mine['mean'], iteration)
            writer.add_scalar("diag/iter_3r_ch1_target", _opp['mean'], iteration)
            iter_3r_sign = stats.get('iter_3r_sign')
            if iter_3r_sign:
                print(f"  Diag[SIGN-iter]: ch0_3row: +:{iter_3r_sign['pos']} -:{iter_3r_sign['neg']} "
                      f"0:{iter_3r_sign['zero']} (n={iter_3r_sign['n']})")

        writer.add_scalar("self_play/avg_game_length", stats['avg_length'], iteration)
        writer.add_scalar("self_play/wins_p1", stats['wins_p1'], iteration)
        writer.add_scalar("self_play/wins_p2", stats['wins_p2'], iteration)
        writer.add_scalar("self_play/draws", stats['draws'], iteration)
        writer.add_scalar("self_play/p1_win_pct", stats['p1_win_pct'], iteration)
        writer.add_scalar("self_play/buffer_size",
                          sum(1 for s in self.buffer.arr if s is not None), iteration)

        pre_seg = stats.get('pre_seg')
        if pre_seg:
            print(f"  PreSeg: vp_corr={pre_seg['vp_corr']:.3f} overlap={pre_seg['overlap']}/20")
            writer.add_scalar("pre_seg/vp_corr", pre_seg['vp_corr'], iteration)
            writer.add_scalar("pre_seg/overlap_20", pre_seg['overlap'], iteration)

        drift = stats.get('drift')
        if drift:
            _cos_detail = " ".join(
                f"{nm}={drift['per_pos'][nm]:.4f}" for nm in drift['pos_names'])
            print(f"  Diag[DRIFT]: backbone cosine_sim mean={drift['cos_mean']:.4f} "
                  f"min={drift['cos_min']:.4f} | {_cos_detail}")
            writer.add_scalar("drift/bb_cosine_mean", drift['cos_mean'], iteration)
            writer.add_scalar("drift/bb_cosine_min", drift['cos_min'], iteration)
            for nm, val in drift['per_pos'].items():
                writer.add_scalar(f"drift/{nm}_cosine", val, iteration)

        wdelta = stats.get('wdelta')
        if wdelta:
            _wd_parts = [f"rb{bi}={rd:.4f}" for bi, rd in sorted(wdelta.items())]
            print(f"  Diag[WDELTA]: {' '.join(_wd_parts)}")
            for bi, rd in wdelta.items():
                writer.add_scalar(f"wdelta/rb{bi}_rel", rd, iteration)

    def _log_mcts_perf(self, iteration):
        """Log MCTS and NN inference performance metrics."""
        if not (hasattr(self, '_batched') and hasattr(self._batched, 'perf')):
            return
        writer = self.writer
        perf = self._batched.perf
        avg_batch = perf["sample_count"] / max(perf["batch_count"], 1)
        writer.add_scalar("perf/mcts_select_expand", perf["select_expand_time"], iteration)
        writer.add_scalar("perf/mcts_backup", perf["backup_time"], iteration)
        writer.add_scalar("perf/nn_time", perf["nn_time"], iteration)
        writer.add_scalar("perf/nn_preprocess", perf["preprocess_time"], iteration)
        writer.add_scalar("perf/nn_transfer", perf["transfer_time"], iteration)
        writer.add_scalar("perf/nn_forward", perf["forward_time"], iteration)
        writer.add_scalar("perf/nn_result", perf["result_time"], iteration)
        writer.add_scalar("perf/nn_postprocess", perf["postprocess_time"], iteration)
        writer.add_scalar("perf/batch_count", perf["batch_count"], iteration)
        writer.add_scalar("perf/avg_batch_size", avg_batch, iteration)
        writer.add_scalar("perf/terminal_hits", perf["terminal_hits"], iteration)
        print(f"  MCTS: select={perf['select_expand_time']:.1f}s "
              f"backup={perf['backup_time']:.1f}s "
              f"terminal_hits={perf['terminal_hits']}")
        print(f"  NN:   forward={perf['forward_time']:.1f}s "
              f"result={perf['result_time']:.1f}s "
              f"preprocess={perf['preprocess_time']:.1f}s "
              f"transfer={perf['transfer_time']:.1f}s | "
              f"batches={perf['batch_count']} "
              f"batch_sz={perf['min_batch']}/{avg_batch:.0f}/{perf['max_batch']}")
        enc_errs = perf.get("encoding_errors", 0)
        enc_total = perf.get("encoding_checks", 0)
        enc_time = perf.get("encoding_time", 0)
        enc_sampled = enc_total // 50 if enc_total else 0
        if enc_errs > 0:
            print(f"  [WARNING] Encoding errors: {enc_errs}/{enc_sampled} sampled "
                  f"({enc_total} total) {enc_time:.2f}s")
        else:
            print(f"  Encoding: {enc_sampled}/{enc_total} sampled, all OK, {enc_time:.2f}s")
        hist = perf.get("batch_histogram", [0]*5)
        print(f"  BatchHist: [1-4]={hist[0]} [5-16]={hist[1]} "
              f"[17-32]={hist[2]} [33-64]={hist[3]} [65+]={hist[4]}")
        apm = perf.get("active_per_move", [])
        if apm:
            print(f"  ActiveGames: min={min(apm)} avg={np.mean(apm):.1f} "
                  f"max={max(apm)} steps={len(apm)}")
        accum = perf.get("accum_rounds", 0)
        if accum > 0:
            print(f"  Accumulation: {accum} move-steps used batch accumulation")
        tr_count = perf.get("tree_reuse_count", 0)
        tr_fresh = perf.get("tree_reuse_fresh_count", 0)
        tr_total = tr_count + tr_fresh
        if tr_total > 0:
            tr_pct = tr_count / tr_total
            tr_avg_v = perf.get("tree_reuse_avg_visits", 0)
            print(f"  TreeReuse: {tr_count}/{tr_total} ({tr_pct:.0%}) "
                  f"avg_reused_visits={tr_avg_v:.1f}")
            writer.add_scalar("perf/tree_reuse_pct", tr_pct, iteration)
            writer.add_scalar("perf/tree_reuse_avg_visits", tr_avg_v, iteration)
        resign_count = perf.get("resign_count", 0)
        resign_checks = perf.get("resign_check_count", 0)
        resign_fp = perf.get("resign_false_positives", 0)
        if resign_count > 0 or resign_checks > 0:
            resign_avg_move = perf.get("resign_avg_move", 0)
            total_games = self.games_per_iteration
            resign_pct = resign_count / max(total_games, 1)
            fp_str = f"false_pos={resign_fp}/{resign_checks}" if resign_checks > 0 else "no_checks"
            print(f"  Resign: {resign_count}/{total_games} ({resign_pct:.0%}) "
                  f"avg_move={resign_avg_move:.1f} "
                  f"{fp_str}")
            writer.add_scalar("self_play/resign_count", resign_count, iteration)
            writer.add_scalar("self_play/resign_pct", resign_pct, iteration)
            writer.add_scalar("self_play/resign_avg_move", resign_avg_move, iteration)
            if resign_checks > 0:
                fp_rate = resign_fp / resign_checks
                writer.add_scalar("self_play/resign_fp_rate", fp_rate, iteration)
        imm_win_n = perf.get("imm_win_count", 0)
        imm_win_frac = perf.get("imm_win_frac", 0)
        imm_win_total = imm_win_n / max(imm_win_frac, 1e-9) if imm_win_frac > 0 else 0
        print(f"  ImmWin: {imm_win_n}/{int(imm_win_total)} ({imm_win_frac:.1%}) positions have immediate winning move")
        writer.add_scalar("self_play/imm_win_frac", imm_win_frac, iteration)
        writer.add_scalar("self_play/imm_win_count", imm_win_n, iteration)

    def _log_training_perf(self, iteration):
        """Log training data prep and gradient computation times."""
        if not hasattr(self, '_train_perf'):
            return
        writer = self.writer
        tp = self._train_perf
        writer.add_scalar("perf/train_data_prep", tp["data_prep_time"], iteration)
        writer.add_scalar("perf/train_gradient", tp["gradient_time"], iteration)
        writer.add_scalar("perf/train_num_samples", tp["num_samples"], iteration)
        writer.add_scalar("perf/train_num_batches", tp["num_batches"], iteration)
        print(f"  Train: data={tp['data_prep_time']:.1f}s "
              f"grad={tp['gradient_time']:.1f}s | "
              f"samples={tp['num_samples']} "
              f"batches={tp['num_batches']}")

    def _log_training_diagnostics(self, iteration, stats):
        """Log all training diagnostic metrics to console and TensorBoard."""
        if not hasattr(self, '_train_diag'):
            return
        writer = self.writer
        d = self._train_diag
        p1_win_pct = stats['p1_win_pct']
        avg_length = stats['avg_length']
        min_length, max_length = stats['min_length'], stats['max_length']

        # TensorBoard scalars
        writer.add_scalar("diag/val_target_mean", d["val_target_mean"], iteration)
        writer.add_scalar("diag/val_target_std", d["val_target_std"], iteration)
        writer.add_scalar("diag/frac_xwins", d["frac_neg"], iteration)
        writer.add_scalar("diag/frac_owins", d["frac_pos"], iteration)
        writer.add_scalar("diag/frac_draws", d["frac_draw"], iteration)
        writer.add_scalar("diag/effective_epochs", d["effective_epochs"], iteration)
        writer.add_scalar("diag/early_vloss", d["early_vloss"], iteration)
        writer.add_scalar("diag/late_vloss", d["late_vloss"], iteration)
        writer.add_scalar("diag/early_ploss", d["early_ploss"], iteration)
        writer.add_scalar("diag/late_ploss", d["late_ploss"], iteration)
        writer.add_scalar("diag/buffer_fill", d["buffer_fill"], iteration)
        writer.add_scalar("diag/pred_v_mean", d["pred_v_mean"], iteration)
        writer.add_scalar("diag/pred_v_std", d["pred_v_std"], iteration)
        writer.add_scalar("diag/pred_v_abs_mean", d["pred_v_abs_mean"], iteration)
        writer.add_scalar("diag/policy_grad_frac", d["policy_grad_frac"], iteration)
        writer.add_scalar("diag/val_loss_floor", d["val_loss_floor"], iteration)
        writer.add_scalar("diag/value_grad_norm", d["avg_value_grad_norm"], iteration)
        writer.add_scalar("diag/policy_grad_norm", d["avg_policy_grad_norm"], iteration)
        writer.add_scalar("diag/policy_entropy", d["policy_entropy"], iteration)
        writer.add_scalar("diag/policy_top1_acc", d["policy_top1_acc"], iteration)
        writer.add_scalar("diag/policy_top3_acc", d["policy_top3_acc"], iteration)
        writer.add_scalar("diag/value_confidence_acc", d["value_confidence_acc"], iteration)
        writer.add_scalar("diag/value_confident_frac", d["value_confident_frac"], iteration)
        for rb_i, rb_gn in d.get("rb_grad_norms", {}).items():
            writer.add_scalar(f"diag/rb{rb_i}_grad_norm", rb_gn, iteration)
        if d.get("vloss_delta") is not None:
            writer.add_scalar("diag/vloss_delta", d["vloss_delta"], iteration)
            writer.add_scalar("diag/pre_train_vloss", d["pre_train_vloss"], iteration)
        writer.add_scalar("diag/phase_vloss_early", d["phase_vloss_early"], iteration)
        writer.add_scalar("diag/phase_vloss_mid", d["phase_vloss_mid"], iteration)
        writer.add_scalar("diag/phase_vloss_late", d["phase_vloss_late"], iteration)
        writer.add_scalar("diag/policy_loss_decisive", d["policy_loss_decisive"], iteration)
        writer.add_scalar("diag/policy_loss_ambiguous", d["policy_loss_ambiguous"], iteration)
        writer.add_scalar("diag/decisive_frac", d["decisive_frac"], iteration)

        # Console output
        print(f"  Diag: targets mean={d['val_target_mean']:+.3f} "
              f"std={d['val_target_std']:.3f} | "
              f"X={d['frac_neg']:.1%} O={d['frac_pos']:.1%} "
              f"draw={d['frac_draw']:.1%}")
        overfit_gap = d['val_vloss'] - d['late_vloss']
        writer.add_scalar("diag/overfit_gap_vloss", overfit_gap, iteration)
        vloss_delta_str = f" delta={d['vloss_delta']:+.4f}" if d.get('vloss_delta') is not None else ""
        print(f"  Diag: eff_epochs={d['effective_epochs']:.1f} "
              f"vlw={d.get('effective_vlw',1.0):.2f} "
              f"steps={d['num_steps']} | "
              f"vloss train={d['late_vloss']:.4f} "
              f"val={d['val_vloss']:.4f} "
              f"(gap={overfit_gap:+.4f}){vloss_delta_str} | "
              f"buf={d['buffer_fill']}/{d['buffer_capacity']}"
              f"{' FULL' if d['buffer_full'] else ''}")
        writer.add_scalar("diag/val_vloss", d["val_vloss"], iteration)
        writer.add_scalar("diag/effective_vlw", d.get("effective_vlw", 1.0), iteration)
        writer.add_scalar("diag/val_ploss", d["val_ploss"], iteration)
        print(f"  Diag: pred_v mean={d['pred_v_mean']:+.3f} "
              f"std={d['pred_v_std']:.3f} "
              f"|v|={d['pred_v_abs_mean']:.3f} | "
              f"policy_grad={d['policy_grad_frac']:.1%} | "
              f"vloss_floor={d['val_loss_floor']:.4f}")
        print(f"  Diag: grad_norms value={d['avg_value_grad_norm']:.4f} "
              f"policy={d['avg_policy_grad_norm']:.4f} "
              f"ratio={d['avg_value_grad_norm']/max(d['avg_policy_grad_norm'],1e-8):.2f}")
        print(f"  Diag: p1_win={p1_win_pct:.1%} "
              f"game_len={avg_length:.1f} ({min_length}-{max_length})")
        print(f"  Diag[P]: entropy={d['policy_entropy']:.3f} "
              f"top1_acc={d['policy_top1_acc']:.1%} "
              f"top3_acc={d['policy_top3_acc']:.1%}")
        print(f"  Diag[P2]: ploss_decisive={d['policy_loss_decisive']:.4f} "
              f"ploss_ambiguous={d['policy_loss_ambiguous']:.4f} "
              f"decisive_frac={d['decisive_frac']:.1%}")
        print(f"  Diag[C]: confident_acc={d['value_confidence_acc']:.1%} "
              f"(frac_confident={d['value_confident_frac']:.1%})")
        rb_gn = d.get('rb_grad_norms', {})
        if rb_gn:
            rb_items = sorted((i, n) for i, n in rb_gn.items() if isinstance(i, int))
            rb_str = " ".join(f"rb{i}={n:.4f}" for i, n in rb_items)
            eff_lr_items = sorted((k, v) for k, v in rb_gn.items() if isinstance(k, str) and 'eff_lr' in k)
            eff_lr_str = " ".join(f"rb{k.replace('_eff_lr','')}={v:.5f}" for k, v in eff_lr_items)
            line = f"  Diag[RB]: grad_norms: {rb_str}"
            if eff_lr_str:
                line += f" | c2_eff_lr: {eff_lr_str}"
            print(line)
        vh_bins = d.get('val_hist', [])
        if vh_bins:
            print(f"  Diag[TH]: targets [-1,-0.5)={vh_bins[0]:.1%} "
                  f"[-0.5,0)={vh_bins[1]:.1%} [0]={vh_bins[2]:.1%} "
                  f"(0,0.5]={vh_bins[3]:.1%} (0.5,1]={vh_bins[4]:.1%}")
        tr = d.get('three_r_diag', {})
        if tr:
            mine = tr.get('mine', {})
            opp = tr.get('opp', {})
            print(f"  Diag[3R]: ch0_3row: n={mine.get('count',0)} target={mine.get('mean_target',0):+.3f} "
                  f"({mine.get('frac',0):.1%}) | "
                  f"ch1_3row: n={opp.get('count',0)} target={opp.get('mean_target',0):+.3f} "
                  f"({opp.get('frac',0):.1%})")
            writer.add_scalar("diag/three_r_ch0_target", mine.get('mean_target', 0), iteration)
            writer.add_scalar("diag/three_r_ch1_target", opp.get('mean_target', 0), iteration)
            writer.add_scalar("diag/three_r_ch0_frac", mine.get('frac', 0), iteration)
            writer.add_scalar("diag/three_r_ch1_frac", opp.get('frac', 0), iteration)
        print(f"  Diag[A]: X_vloss={d['x_vloss']:.4f} O_vloss={d['o_vloss']:.4f} | "
              f"X_target={d['x_target_mean']:+.3f} O_target={d['o_target_mean']:+.3f} | "
              f"X_pred={d['x_pred_mean']:+.3f} O_pred={d['o_pred_mean']:+.3f}")
        print(f"  Diag[BufBias]: X-to-move: n={d.get('buf_n_x',0)} "
              f"mean_tgt={d.get('buf_mean_tgt_x',0):+.3f} "
              f"frac_pos={d.get('buf_frac_pos_x',0):.1%} | "
              f"O-to-move: n={d.get('buf_n_o',0)} "
              f"mean_tgt={d.get('buf_mean_tgt_o',0):+.3f} "
              f"frac_pos={d.get('buf_frac_pos_o',0):.1%}")
        writer.add_scalar("pbias/buf_mean_tgt_x", d.get('buf_mean_tgt_x', 0), iteration)
        writer.add_scalar("pbias/buf_mean_tgt_o", d.get('buf_mean_tgt_o', 0), iteration)
        writer.add_scalar("pbias/buf_frac_pos_x", d.get('buf_frac_pos_x', 0), iteration)
        writer.add_scalar("pbias/buf_frac_pos_o", d.get('buf_frac_pos_o', 0), iteration)
        print(f"  Diag[PBias]: pre: X_pred={d.get('pbias_pre_x_pred',0):+.3f} "
              f"X_acc={d.get('pbias_pre_x_acc',0):.1%} | "
              f"O_pred={d.get('pbias_pre_o_pred',0):+.3f} "
              f"O_acc={d.get('pbias_pre_o_acc',0):.1%}")
        print(f"  Diag[PBias]: post: X_pred={d.get('pbias_post_x_pred',0):+.3f} "
              f"X_acc={d.get('pbias_post_x_acc',0):.1%} | "
              f"O_pred={d.get('pbias_post_o_pred',0):+.3f} "
              f"O_acc={d.get('pbias_post_o_acc',0):.1%}")
        writer.add_scalar("pbias/pre_x_pred", d.get('pbias_pre_x_pred', 0), iteration)
        writer.add_scalar("pbias/pre_o_pred", d.get('pbias_pre_o_pred', 0), iteration)
        writer.add_scalar("pbias/post_x_pred", d.get('pbias_post_x_pred', 0), iteration)
        writer.add_scalar("pbias/post_o_pred", d.get('pbias_post_o_pred', 0), iteration)
        writer.add_scalar("pbias/pre_x_acc", d.get('pbias_pre_x_acc', 0), iteration)
        writer.add_scalar("pbias/pre_o_acc", d.get('pbias_pre_o_acc', 0), iteration)
        writer.add_scalar("pbias/post_x_acc", d.get('pbias_post_x_acc', 0), iteration)
        writer.add_scalar("pbias/post_o_acc", d.get('pbias_post_o_acc', 0), iteration)
        pc = d.get('phase_counts', {})
        print(f"  Diag[GP]: vloss early={d['phase_vloss_early']:.4f}({pc.get('early',0)}) "
              f"mid={d['phase_vloss_mid']:.4f}({pc.get('mid',0)}) "
              f"late={d['phase_vloss_late']:.4f}({pc.get('late',0)})")
        gs = d.get('grad_stats', {})
        if gs:
            err_trend = gs.get('error_mean_trend', [])
            trend_str = ""
            if len(err_trend) >= 2:
                trend_str = f" err_trend={err_trend[0]:+.3f}->{err_trend[-1]:+.3f}"
            print(f"  Diag[F]: fc1_grad={gs.get('fc1_grad_norm_mean',0):.4f} "
                  f"fc2_grad={gs.get('fc2_grad_norm_mean',0):.4f} | "
                  f"fc1_mean={gs.get('fc1_grad_mean',0):+.6f} "
                  f"fc2_mean={gs.get('fc2_grad_mean',0):+.6f}"
                  f"{trend_str}")

        vh = d.get('vh_diag', {})
        if vh:
            self._log_value_head_diagnostics(vh, d, iteration)

    def _log_value_head_diagnostics(self, vh, d, iteration):
        """Log value head health diagnostics to console and TensorBoard."""
        writer = self.writer
        print(f"  Diag[V]: dead={vh['dead_neurons']} "
              f"active={vh.get('active_neurons','?')}/{vh['total_neurons']} "
              f"| WDL: entropy={vh['wdl_entropy']:.3f} "
              f"conf={vh['wdl_confidence']:.3f} "
              f"acc={vh['wdl_accuracy']:.1%} "
              f"v={vh['wdl_scalar_mean']:+.3f}±{vh['wdl_scalar_std']:.3f}")
        print(f"  Diag[V1]: WDL probs: W={vh['wdl_win_prob']:.3f} "
              f"D={vh['wdl_draw_prob']:.3f} L={vh['wdl_loss_prob']:.3f} | "
              f"logit_std={vh['wdl_logit_std']:.3f} "
              f"logit_range={vh['wdl_logit_range']:.3f}")
        print(f"  Diag[V2]: fc2_w=[{vh['fc2_w_min']:+.3f},{vh['fc2_w_max']:+.3f}] "
              f"norm={vh['fc2_w_norm']:.3f} "
              f"bias=[W:{vh['fc2_bias_w']:+.3f} D:{vh['fc2_bias_d']:+.3f} L:{vh['fc2_bias_l']:+.3f}] | "
              f"fc1_w_norm={vh['fc1_w_norm']:.3f} | "
              f"backbone std={vh['backbone_std']:.3f} |x|={vh['backbone_abs_mean']:.3f}")
        # Metric 2: activation percentiles
        print(f"  Diag[V4]: fc1_act p10={vh['fc1_act_p10']:.4f} "
              f"p50={vh['fc1_act_p50']:.4f} "
              f"p90={vh['fc1_act_p90']:.4f}")
        # Metric 3: value conv channels
        ch_str = " ".join(f"ch{i}={vh['vconv_ch_abs_mean'][i]:.3f}"
                          for i in range(vh['vconv_n_channels']))
        print(f"  Diag[V5]: vconv_channels: {ch_str} "
              f"dead_ch={vh['vconv_dead_channels']}/{vh['vconv_n_channels']}")
        # Metric 4: neuron death tracking
        dead_ids = vh.get('dead_neuron_ids', [])
        weak = list(zip(vh['weakest_5_ids'], vh['weakest_5_vals']))
        print(f"  Diag[V6]: dead_ids={dead_ids[:10]}"
              f"{'...' if len(dead_ids) > 10 else ''} | "
              f"weakest={weak}")
        # Metric 5: weight growth rate
        print(f"  Diag[V7]: fc2_w_d={vh['fc2_w_norm_delta']:+.4f} "
              f"fc1_w_d={vh['fc1_w_norm_delta']:+.4f}")
        # Metric 1: gradient flow to dead vs alive
        print(f"  Diag[V8]: grad_flow "
              f"dead={vh['grad_dead_mean']:.6f} "
              f"alive={vh['grad_alive_mean']:.6f} "
              f"ratio={vh['grad_dead_mean']/max(vh['grad_alive_mean'],1e-10):.3f}")
        # Value conv decay chain
        print(f"  Diag[VC]: backbone_raw |x|={vh['backbone_raw_abs']:.3f} "
              f"std={vh['backbone_raw_std']:.3f} | "
              f"vconv_w norm={vh['vc_w_norm']:.3f} |w|={vh['vc_w_abs_mean']:.4f}")
        print(f"  Diag[VC2]: pre_bn |x|={vh['vconv_pre_bn_abs']:.3f} "
              f"std={vh['vconv_pre_bn_std']:.3f} | "
              f"post_bn |x|={vh['vbn_post_abs']:.3f} "
              f"std={vh['vbn_post_std']:.3f} | "
              f"bn_ratio={vh['bn_ratio']:.3f}")
        gamma = vh['vbn_gamma']
        beta = vh['vbn_beta']
        rvar = vh['vbn_running_var']
        gamma_str = " ".join(f"{g:.3f}" for g in gamma)
        beta_str = " ".join(f"{b:+.3f}" for b in beta)
        rvar_str = " ".join(f"{v:.3f}" for v in rvar)
        print(f"  Diag[VC3]: bn_gamma=[{gamma_str}] "
              f"min={vh['vbn_gamma_min']:.3f}")
        print(f"  Diag[VC4]: bn_beta=[{beta_str}]")
        print(f"  Diag[VC5]: bn_run_var=[{rvar_str}]")
        print(f"  Diag[VC6]: policy_conv |x|={vh['pconv_abs']:.3f} "
              f"std={vh['pconv_std']:.3f} | "
              f"vconv_grad={vh['vconv_grad_norm']:.4f}")
        # Backbone per-channel stats
        if 'bb_n_channels' in vh:
            print(f"  Diag[BB]: backbone {vh['bb_n_channels']}ch "
                  f"dead={vh['bb_dead_channels']} | "
                  f"|x| p10={vh['bb_ch_p10']:.4f} "
                  f"p50={vh['bb_ch_p50']:.4f} "
                  f"p90={vh['bb_ch_p90']:.4f} "
                  f"max={vh['bb_ch_max']:.4f}")
            print(f"  Diag[BB1]: top5={vh['bb_top5']} "
                  f"bot5={vh['bb_bot5']}")
        # Backbone gradient decomposition
        if 'bb_v_grad_norm' in vh:
            evlw = d.get('effective_vlw', 1.0)
            v_eff = vh['bb_v_grad_norm'] * evlw
            eff_ratio = v_eff / max(vh['bb_p_grad_norm'], 1e-10)
            print(f"  Diag[BB2]: bb_grad output: "
                  f"v={vh['bb_v_grad_norm']:.4f} "
                  f"p={vh['bb_p_grad_norm']:.4f} "
                  f"ratio={vh['bb_grad_ratio']:.3f} "
                  f"(eff_v={v_eff:.4f} eff_ratio={eff_ratio:.3f})")
            print(f"  Diag[BB3]: ch_dominance: "
                  f"value={vh['bb_n_value_dom']}/{vh.get('bb_n_channels',256)} "
                  f"policy={vh['bb_n_policy_dom']}/{vh.get('bb_n_channels',256)} | "
                  f"top_v={vh['bb_top_v_channels'][:3]} "
                  f"top_p={vh['bb_top_p_channels'][:3]}")
        if 'bb_param_v_grad' in vh:
            evlw = d.get('effective_vlw', 1.0)
            pv_eff = vh['bb_param_v_grad'] * evlw
            pv_eff_ratio = pv_eff / max(vh['bb_param_p_grad'], 1e-10)
            rb = vh.get('bb_res_block_grads', {})
            rb_str = " ".join(
                f"rb{k}: v={v[0]:.3f} p={v[1]:.3f} r={v[2]:.2f}"
                for k, v in sorted(rb.items()))
            print(f"  Diag[BB4]: bb_grad params: "
                  f"v={vh['bb_param_v_grad']:.4f} "
                  f"p={vh['bb_param_p_grad']:.4f} "
                  f"ratio={vh['bb_param_grad_ratio']:.3f} "
                  f"(eff_v={pv_eff:.4f} eff_ratio={pv_eff_ratio:.3f})")
            if rb_str:
                print(f"  Diag[BB5]: {rb_str}")
            # (#10) Value gradient survival
            vgs = vh.get('rb_v_grad_survival', {})
            if vgs:
                vgs_str = " ".join(f"rb{k}={v:.3f}" for k, v in sorted(vgs.items()))
                print(f"  Diag[BB7]: v_grad_survival: {vgs_str}")
        # (#1) Backbone gradient conflict
        if 'bb_grad_cosine_sim' in vh:
            print(f"  Diag[BB6]: grad_cosine={vh['bb_grad_cosine_sim']:+.3f} "
                  f"conflict_ch={vh['bb_grad_conflict_channels']} "
                  f"aligned_ch={vh['bb_grad_aligned_channels']}")
            writer.add_scalar("diag/bb_grad_cosine_sim", vh['bb_grad_cosine_sim'], iteration)
            writer.add_scalar("diag/bb_grad_conflict_channels", vh['bb_grad_conflict_channels'], iteration)
        if 'vp_weight_corr' in vh:
            print(f"  Diag[VP]: weight_corr={vh['vp_weight_corr']:.3f} "
                  f"overlap_top20={vh['vp_overlap_20']}/20 | "
                  f"act: val_top20={vh['vp_val_top20_act']:.4f} "
                  f"pol_top20={vh['vp_pol_top20_act']:.4f} "
                  f"ratio={vh['vp_val_top20_act']/max(vh['vp_pol_top20_act'],1e-10):.2f}")
            print(f"  Diag[VP2]: health_corr: "
                  f"val={vh['vp_val_health_corr']:.3f} "
                  f"pol={vh['vp_pol_health_corr']:.3f}")
        if 'svd_bb_rank90' in vh:
            print(f"  Diag[SVD]: backbone rb{len(self.net.res_blocks)-1}.conv2: "
                  f"rank90={vh['svd_bb_rank90']}/{vh['svd_bb_total']} "
                  f"rank99={vh['svd_bb_rank99']}/{vh['svd_bb_total']} "
                  f"near_zero_sv={vh['svd_bb_near_zero']} "
                  f"bn_dead={vh['bn_dead_deepest']}")
            pc_str = " ".join(f"ch{i}={n:.3f}" for i, n in enumerate(vh['pconv_ch_norms']))
            print(f"  Diag[PH]: policy_fc: "
                  f"rank90={vh['svd_pfc_rank90']}/{vh['svd_pfc_max_rank']} "
                  f"rank99={vh['svd_pfc_rank99']}/{vh['svd_pfc_max_rank']} | "
                  f"pconv: {pc_str}")
        # Per-block BN and activation diagnostics
        all_rb_bn_data = vh.get('all_rb_bn', {})
        rb_act_data = vh.get('rb_act_stats', {})
        for bi in sorted(set(list(all_rb_bn_data.keys()) + list(rb_act_data.keys()))):
            parts = [f"  Diag[RB{bi}]:"]
            if bi in all_rb_bn_data:
                rbd = all_rb_bn_data[bi]
                parts.append(f"bn2: dead={rbd.get('dead',0)} "
                             f"neg_gamma={rbd.get('neg_gamma',0)} "
                             f"eff_gain={rbd.get('eff_gain_mean',0):.2f}/"
                             f"{rbd.get('eff_gain_max',0):.2f} "
                             f"gamma={rbd.get('gamma_mean',0):.3f}(+/-{rbd.get('gamma_std',0):.3f}) "
                             f"sqrt_var={rbd.get('sqrt_var_mean',0):.3f}(+/-{rbd.get('sqrt_var_std',0):.3f})")
                parts.append(f"svd_rank90={rbd.get('svd_rank90',0)}/"
                             f"{rbd.get('svd_total',0)}")
            if bi in rb_act_data:
                rad = rb_act_data[bi]
                parts.append(f"|x|={rad['abs_mean']:.3f} "
                             f"std={rad['std']:.3f} "
                             f"dead={rad['dead_channels']}")
            print(" | ".join(parts))
            # Second line: conv weight norms, conv2 raw output (residual branch), bn2 stats, res rank, ch dominance
            parts2 = []
            rcn = vh.get('rb_conv_norms', {}).get(bi)
            if rcn:
                parts2.append(f"w: c1={rcn['conv1']:.3f} c2={rcn['conv2']:.3f} c2_d={rcn.get('c2_delta',0):+.3f}")
            rbs = vh.get('rb_bn2_stats', {}).get(bi)
            if rbs:
                s = ""
                if 'bn2_out_abs' in rbs:
                    s += f"bn2_out: |x|={rbs['bn2_out_abs']:.3f} std={rbs['bn2_out_std']:.3f}"
                if 'conv2_raw_var' in rbs:
                    if s: s += " | "
                    s += f"conv2_raw: var={rbs['conv2_raw_var']:.4f} |x|={rbs['conv2_raw_abs']:.3f}"
                if 'bn2_batch_vs_run_var_mean' in rbs:
                    if s: s += " | "
                    s += (f"bv/rv={rbs['bn2_batch_vs_run_var_mean']:.3f}"
                          f"(+/-{rbs['bn2_batch_vs_run_var_std']:.3f})"
                          f" bv={rbs['bn2_batch_var_mean']:.4f}"
                          f" rv={rbs['bn2_run_var_mean']:.4f}")
                if s:
                    parts2.append(s)
            rrk = vh.get('rb_res_rank', {}).get(bi)
            if rrk:
                parts2.append(f"res_rank90={rrk['rank90']}/{rrk['total']}")
            rcd = vh.get('rb_ch_dominance', {})
            if bi in rcd:
                parts2.append(f"ch_vdom={rcd[bi]}/{vh.get('bb_n_channels', '?')}")
            if parts2:
                print(f"           {' | '.join(parts2)}")
        # (#4) final_bn gamma tracking
        if 'final_bn_eff_gain_mean' in vh:
            print(f"  Diag[FBN]: eff_gain "
                  f"mean={vh['final_bn_eff_gain_mean']:.3f} "
                  f"min={vh['final_bn_eff_gain_min']:.3f} "
                  f"max={vh['final_bn_eff_gain_max']:.3f} "
                  f"dead={vh['final_bn_dead']} "
                  f"| gamma={vh.get('final_bn_gamma_mean',0):.3f}(+/-{vh.get('final_bn_gamma_std',0):.3f}) "
                  f"sqrt_var={vh.get('final_bn_sqrt_var_mean',0):.3f}(+/-{vh.get('final_bn_sqrt_var_std',0):.3f})")
            writer.add_scalar("diag/final_bn_eff_gain_mean", vh['final_bn_eff_gain_mean'], iteration)
            writer.add_scalar("diag/final_bn_eff_gain_min", vh['final_bn_eff_gain_min'], iteration)
            writer.add_scalar("diag/final_bn_dead", vh['final_bn_dead'], iteration)
            writer.add_scalar("diag/final_bn_gamma_mean", vh.get('final_bn_gamma_mean', 0), iteration)
            writer.add_scalar("diag/final_bn_gamma_std", vh.get('final_bn_gamma_std', 0), iteration)
            writer.add_scalar("diag/final_bn_sqrt_var_mean", vh.get('final_bn_sqrt_var_mean', 0), iteration)
            writer.add_scalar("diag/final_bn_sqrt_var_std", vh.get('final_bn_sqrt_var_std', 0), iteration)
        # (#5) Residual contribution ratio
        rr = vh.get('rb_residual_ratios', {})
        if rr:
            rr_str = " ".join(f"rb{k}={v:.3f}" for k, v in sorted(rr.items()))
            print(f"  Diag[RR]: residual_ratio: {rr_str}")
            for bi, ratio in rr.items():
                writer.add_scalar(f"rb{bi}/residual_ratio", ratio, iteration)
        # (#3) Initial conv weight norm
        if 'init_conv_w_norm' in vh:
            print(f"  Diag[IC]: conv_w norm={vh['init_conv_w_norm']:.4f} "
                  f"|w|={vh['init_conv_w_abs_mean']:.5f} "
                  f"delta={vh['init_conv_w_norm_delta']:+.4f}")
            writer.add_scalar("diag/init_conv_w_norm", vh['init_conv_w_norm'], iteration)
        # Tensorboard
        writer.add_scalar("vh/dead_neurons", vh["dead_neurons"], iteration)
        writer.add_scalar("vh/active_neurons", vh.get("active_neurons", 0), iteration)
        # WDL metrics (replaces pre_tanh/saturated)
        writer.add_scalar("vh/wdl_entropy", vh["wdl_entropy"], iteration)
        writer.add_scalar("vh/wdl_confidence", vh["wdl_confidence"], iteration)
        writer.add_scalar("vh/wdl_accuracy", vh["wdl_accuracy"], iteration)
        writer.add_scalar("vh/wdl_logit_std", vh["wdl_logit_std"], iteration)
        writer.add_scalar("vh/wdl_logit_range", vh["wdl_logit_range"], iteration)
        writer.add_scalar("vh/wdl_win_prob", vh["wdl_win_prob"], iteration)
        writer.add_scalar("vh/wdl_draw_prob", vh["wdl_draw_prob"], iteration)
        writer.add_scalar("vh/wdl_loss_prob", vh["wdl_loss_prob"], iteration)
        writer.add_scalar("vh/wdl_scalar_mean", vh["wdl_scalar_mean"], iteration)
        writer.add_scalar("vh/wdl_scalar_std", vh["wdl_scalar_std"], iteration)
        writer.add_scalar("vh/fc2_w_max", vh["fc2_w_max"], iteration)
        writer.add_scalar("vh/fc2_w_min", vh["fc2_w_min"], iteration)
        writer.add_scalar("vh/fc2_w_norm", vh["fc2_w_norm"], iteration)
        writer.add_scalar("vh/fc1_w_norm", vh["fc1_w_norm"], iteration)
        writer.add_scalar("vh/backbone_std", vh["backbone_std"], iteration)
        writer.add_scalar("vh/fc1_act_p10", vh["fc1_act_p10"], iteration)
        writer.add_scalar("vh/fc1_act_p50", vh["fc1_act_p50"], iteration)
        writer.add_scalar("vh/fc1_act_p90", vh["fc1_act_p90"], iteration)
        writer.add_scalar("vh/vconv_dead_channels", vh["vconv_dead_channels"], iteration)
        writer.add_scalar("vh/fc2_w_norm_delta", vh["fc2_w_norm_delta"], iteration)
        writer.add_scalar("vh/fc1_w_norm_delta", vh["fc1_w_norm_delta"], iteration)
        writer.add_scalar("vh/grad_dead_mean", vh["grad_dead_mean"], iteration)
        writer.add_scalar("vh/grad_alive_mean", vh["grad_alive_mean"], iteration)
        writer.add_scalar("vc/backbone_raw_abs", vh["backbone_raw_abs"], iteration)
        writer.add_scalar("vc/vconv_pre_bn_abs", vh["vconv_pre_bn_abs"], iteration)
        writer.add_scalar("vc/vbn_post_abs", vh["vbn_post_abs"], iteration)
        writer.add_scalar("vc/bn_ratio", vh["bn_ratio"], iteration)
        writer.add_scalar("vc/vc_w_norm", vh["vc_w_norm"], iteration)
        writer.add_scalar("vc/vbn_gamma_min", vh["vbn_gamma_min"], iteration)
        writer.add_scalar("vc/vbn_gamma_mean", vh["vbn_gamma_mean"], iteration)
        writer.add_scalar("vc/pconv_abs", vh["pconv_abs"], iteration)
        writer.add_scalar("vc/vconv_grad_norm", vh["vconv_grad_norm"], iteration)
        # Backbone per-channel stats
        if 'bb_n_channels' in vh:
            writer.add_scalar("bb/dead_channels", vh["bb_dead_channels"], iteration)
            writer.add_scalar("bb/ch_p10", vh["bb_ch_p10"], iteration)
            writer.add_scalar("bb/ch_p50", vh["bb_ch_p50"], iteration)
            writer.add_scalar("bb/ch_p90", vh["bb_ch_p90"], iteration)
        # Backbone gradient decomposition
        if 'bb_v_grad_norm' in vh:
            writer.add_scalar("bb/v_grad_norm", vh["bb_v_grad_norm"], iteration)
            writer.add_scalar("bb/p_grad_norm", vh["bb_p_grad_norm"], iteration)
            writer.add_scalar("bb/grad_ratio", vh["bb_grad_ratio"], iteration)
            writer.add_scalar("bb/n_value_dom", vh["bb_n_value_dom"], iteration)
            writer.add_scalar("bb/n_policy_dom", vh["bb_n_policy_dom"], iteration)
        if 'bb_param_v_grad' in vh:
            writer.add_scalar("bb/param_v_grad", vh["bb_param_v_grad"], iteration)
            writer.add_scalar("bb/param_p_grad", vh["bb_param_p_grad"], iteration)
            writer.add_scalar("bb/param_grad_ratio", vh["bb_param_grad_ratio"], iteration)
        if 'vp_weight_corr' in vh:
            writer.add_scalar("vp/weight_corr", vh["vp_weight_corr"], iteration)
            writer.add_scalar("vp/overlap_20", vh["vp_overlap_20"], iteration)
            writer.add_scalar("vp/val_top20_act", vh["vp_val_top20_act"], iteration)
            writer.add_scalar("vp/pol_top20_act", vh["vp_pol_top20_act"], iteration)
            writer.add_scalar("vp/val_health_corr", vh["vp_val_health_corr"], iteration)
            writer.add_scalar("vp/pol_health_corr", vh["vp_pol_health_corr"], iteration)
        if 'svd_bb_rank90' in vh:
            writer.add_scalar("svd/bb_rank90", vh["svd_bb_rank90"], iteration)
            writer.add_scalar("svd/bb_rank99", vh["svd_bb_rank99"], iteration)
            writer.add_scalar("svd/bb_near_zero", vh["svd_bb_near_zero"], iteration)
            writer.add_scalar("svd/bn_dead_deepest", vh["bn_dead_deepest"], iteration)
            writer.add_scalar("svd/pfc_rank90", vh["svd_pfc_rank90"], iteration)
            writer.add_scalar("svd/pfc_rank99", vh["svd_pfc_rank99"], iteration)
            for i, n in enumerate(vh['pconv_ch_norms']):
                writer.add_scalar(f"ph/pconv_ch{i}_norm", n, iteration)
        # Per-block BN and activation TensorBoard
        for bi, rbd in vh.get('all_rb_bn', {}).items():
            writer.add_scalar(f"rb{bi}/bn2_dead", rbd.get("dead", 0), iteration)
            writer.add_scalar(f"rb{bi}/bn2_neg_gamma", rbd.get("neg_gamma", 0), iteration)
            writer.add_scalar(f"rb{bi}/bn2_eff_gain_mean", rbd.get("eff_gain_mean", 0), iteration)
            writer.add_scalar(f"rb{bi}/svd_rank90", rbd.get("svd_rank90", 0), iteration)
            writer.add_scalar(f"rb{bi}/bn2_gamma_mean", rbd.get("gamma_mean", 0), iteration)
            writer.add_scalar(f"rb{bi}/bn2_sqrt_var_mean", rbd.get("sqrt_var_mean", 0), iteration)
        for bi, rad in vh.get('rb_act_stats', {}).items():
            writer.add_scalar(f"rb{bi}/act_abs_mean", rad["abs_mean"], iteration)
            writer.add_scalar(f"rb{bi}/act_std", rad["std"], iteration)
            writer.add_scalar(f"rb{bi}/act_dead_channels", rad["dead_channels"], iteration)
        # (#6) Per-conv weight norms
        for bi, rcn in vh.get('rb_conv_norms', {}).items():
            writer.add_scalar(f"rb{bi}/conv1_w_norm", rcn["conv1"], iteration)
            writer.add_scalar(f"rb{bi}/conv2_w_norm", rcn["conv2"], iteration)
            writer.add_scalar(f"rb{bi}/conv2_w_delta", rcn.get("c2_delta", 0), iteration)
        # (#7-8) BN2 output scale + pre-BN2 variance
        for bi, rbs in vh.get('rb_bn2_stats', {}).items():
            writer.add_scalar(f"rb{bi}/bn2_out_abs", rbs["bn2_out_abs"], iteration)
            writer.add_scalar(f"rb{bi}/bn2_out_std", rbs["bn2_out_std"], iteration)
            if 'conv2_raw_var' in rbs:
                writer.add_scalar(f"rb{bi}/conv2_raw_var", rbs["conv2_raw_var"], iteration)
                writer.add_scalar(f"rb{bi}/conv2_raw_abs", rbs["conv2_raw_abs"], iteration)
            if 'bn2_batch_vs_run_var_mean' in rbs:
                writer.add_scalar(f"rb{bi}/bn2_bv_rv_ratio", rbs["bn2_batch_vs_run_var_mean"], iteration)
                writer.add_scalar(f"rb{bi}/bn2_batch_var", rbs["bn2_batch_var_mean"], iteration)
                writer.add_scalar(f"rb{bi}/bn2_run_var", rbs["bn2_run_var_mean"], iteration)
        # (#11) Residual path effective rank
        for bi, rrk in vh.get('rb_res_rank', {}).items():
            writer.add_scalar(f"rb{bi}/res_rank90", rrk["rank90"], iteration)
            writer.add_scalar(f"rb{bi}/res_rank99", rrk["rank99"], iteration)
        # (#9) Per-block channel dominance
        for bi, nv in vh.get('rb_ch_dominance', {}).items():
            writer.add_scalar(f"rb{bi}/ch_value_dom", nv, iteration)
        # (#10) Value gradient survival
        for bn, sv in vh.get('rb_v_grad_survival', {}).items():
            writer.add_scalar(f"rb{bn}/v_grad_survival", sv, iteration)

    def _log_selfplay_value_diagnostics(self, iteration):
        """Log self-play value prediction diagnostics."""
        writer = self.writer
        if hasattr(self, '_batched') and hasattr(self._batched, 'value_diag'):
            vd = self._batched.value_diag
            if vd:
                writer.add_scalar("selfplay_diag/mean_nnet_value", vd["mean_nnet_value"], iteration)
                writer.add_scalar("selfplay_diag/std_nnet_value", vd["std_nnet_value"], iteration)
                writer.add_scalar("selfplay_diag/frac_saturated", vd["frac_saturated_any"], iteration)
                writer.add_scalar("selfplay_diag/sign_accuracy", vd["sign_accuracy"], iteration)
                writer.add_scalar("selfplay_diag/mae_vs_outcome", vd["mae_vs_outcome"], iteration)
                writer.add_scalar("selfplay_diag/pred_outcome_corr", vd["pred_outcome_corr"], iteration)
                writer.add_scalar("selfplay_diag/mean_when_x_moves", vd["mean_when_x_moves"], iteration)
                writer.add_scalar("selfplay_diag/mean_when_o_moves", vd["mean_when_o_moves"], iteration)
                print(f"  SelfPlay: nnet_v mean={vd['mean_nnet_value']:+.3f} "
                      f"std={vd['std_nnet_value']:.3f} | "
                      f"hi_conf={vd['frac_saturated_any']:.1%} | "
                      f"sign_acc={vd['sign_accuracy']:.1%} | "
                      f"MAE={vd['mae_vs_outcome']:.3f} | "
                      f"corr={vd['pred_outcome_corr']:+.3f}")
                print(f"  SelfPlay: v_when_X={vd['mean_when_x_moves']:+.3f} "
                      f"v_when_O={vd['mean_when_o_moves']:+.3f} | "
                      f"conf+={vd['frac_saturated_pos']:.1%} "
                      f"conf-={vd['frac_saturated_neg']:.1%}")
                if 'mcts_visit_entropy_mean' in vd:
                    writer.add_scalar("selfplay_diag/mcts_visit_entropy",
                                       vd["mcts_visit_entropy_mean"], iteration)
                    print(f"  SelfPlay: mcts_visit_entropy="
                          f"{vd['mcts_visit_entropy_mean']:.3f} "
                          f"(std={vd['mcts_visit_entropy_std']:.3f})")
                # (#7) MCTS Q vs nnet value agreement
                if 'mcts_nnet_corr' in vd:
                    writer.add_scalar("selfplay_diag/mcts_nnet_corr", vd["mcts_nnet_corr"], iteration)
                    writer.add_scalar("selfplay_diag/mcts_nnet_mae", vd["mcts_nnet_mae"], iteration)
                    writer.add_scalar("selfplay_diag/mcts_correction_mean", vd["mcts_correction_mean"], iteration)
                    print(f"  SelfPlay: mcts_Q mean={vd['mcts_q_mean']:+.3f} "
                          f"std={vd['mcts_q_std']:.3f} | "
                          f"nnet_Q_corr={vd['mcts_nnet_corr']:+.3f} "
                          f"MAE={vd['mcts_nnet_mae']:.3f} "
                          f"correction={vd['mcts_correction_mean']:+.3f}")


    def _log_intra_iteration_dynamics(self, iteration):
        """Log confidence distribution, sub-iteration dynamics, and value trajectory."""
        writer = self.writer
        if hasattr(self, '_train_diag'):
            cd = self._train_diag.get('conf_dist', {})
            if cd:
                print(f"  Diag[CONF]: |v|<0.1={cd.get('very_low',0):.1%} "
                      f"0.1-0.3={cd.get('low',0):.1%} "
                      f"0.3-0.6={cd.get('medium',0):.1%} "
                      f"0.6-0.9={cd.get('high',0):.1%} "
                      f"|v|>0.9={cd.get('very_high',0):.1%}")
                for bk, bv in cd.items():
                    writer.add_scalar(f"conf/{bk}", bv, iteration)

            # --- Sub-iteration dynamics summary ---
            sil = self._train_diag.get('sub_iter_log', [])
            if len(sil) >= 3:
                # Print first, middle, last entries to show intra-iteration trend
                indices = [0, len(sil) // 2, len(sil) - 1]
                parts = []
                for idx in indices:
                    e = sil[idx]
                    parts.append(f"s{e['step']}: vl={e['vloss']:.4f} "
                                 f"pl={e['ploss']:.4f} |v|={e['mean_conf']:.3f} "
                                 f"v={e['mean_v']:+.3f}")
                print(f"  Diag[SUB]: {' -> '.join(parts)}")
                # Log first/last to tensorboard for trend detection
                writer.add_scalar("sub/vloss_first", sil[0]['vloss'], iteration)
                writer.add_scalar("sub/vloss_last", sil[-1]['vloss'], iteration)
                writer.add_scalar("sub/conf_first", sil[0]['mean_conf'], iteration)
                writer.add_scalar("sub/conf_last", sil[-1]['mean_conf'], iteration)
                writer.add_scalar("sub/mean_v_first", sil[0]['mean_v'], iteration)
                writer.add_scalar("sub/mean_v_last", sil[-1]['mean_v'], iteration)

        # --- Intra-iteration value trajectory on FixedEval positions ---
        if hasattr(self, '_train_diag'):
            _traj = self._train_diag.get('fixed_eval_trajectory', [])
            if _traj and len(_traj) >= 2:
                # Print trajectory for each position: step0->step300->...->stepN
                _names = [k for k in _traj[0] if k != 'step']
                for _pn in _names:
                    _vals = [f"s{e['step']}:{e[_pn]:+.3f}" for e in _traj]
                    print(f"  Diag[TRAJ]: {_pn}: {' -> '.join(_vals)}")
                    # TB: log first and last values for trend detection
                    writer.add_scalar(f"traj/{_pn}_first", _traj[0][_pn], iteration)
                    writer.add_scalar(f"traj/{_pn}_last", _traj[-1][_pn], iteration)


    def _eval_diagnostic_positions(self, iteration, prefix="", label="FixedEval"):
        """Evaluate the network on fixed diagnostic positions every iteration.

        This helps track how the value head evolves over training on known positions.
        Args:
            prefix: prefix for tensorboard keys (e.g. "pre_" for pre-training)
            label: label for console output
        """
        self.net.eval()
        positions = self._get_diagnostic_positions()
        if not positions:
            return

        # ENCODING_CHECK: Print raw tensor encoding once (first call only)
        if not getattr(self, '_encoding_checked', False):
            self._encoding_checked = True
            print(f"  Diag[ENC]: Fixed position encoding check:")
            for name, state_input, _ in positions:
                player_val = state_input[2, 0, 0]
                player_str = "X" if player_val < 0 else "O"
                ch0_rows = []
                ch1_rows = []
                for r in range(state_input.shape[1]):
                    ch0_rows.append("".join(
                        "1" if state_input[0, r, c] > 0 else "."
                        for c in range(state_input.shape[2])))
                    ch1_rows.append("".join(
                        "1" if state_input[1, r, c] > 0 else "."
                        for c in range(state_input.shape[2])))
                print(f"    {name}: player={player_val:+.0f} ({player_str}) "
                      f"ch0(me)={'/'.join(ch0_rows)} "
                      f"ch1(opp)={'/'.join(ch1_rows)}")

        print(f"  {label}:")
        for name, state_input, expected_str in positions:
            value, policy = self.net.predict(state_input)
            top_action = np.argmax(policy)
            # Channel-swap: evaluate with ch0↔ch1 swapped to measure channel discrimination
            swapped_input = state_input.copy()
            swapped_input[0], swapped_input[1] = state_input[1].copy(), state_input[0].copy()
            swap_value, _ = self.net.predict(swapped_input)
            swap_delta = value - swap_value
            # SYMMETRY: proper mirror test (swap ch0↔ch1 AND negate ch2)
            # For a symmetric network, V(pos) + V(mirror(pos)) ≈ 0
            mirror_input = state_input.copy()
            mirror_input[0], mirror_input[1] = state_input[1].copy(), state_input[0].copy()
            mirror_input[2] = -state_input[2]  # flip player indicator
            mirror_value, _ = self.net.predict(mirror_input)
            sym_err = value + mirror_value  # should be ~0
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_value", value, iteration)
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_top_action", top_action, iteration)
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_swap_delta", swap_delta, iteration)
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_sym_err", sym_err, iteration)
            print(f"    {name}: V={value:+.4f} top_act={top_action} swap_d={swap_delta:+.4f} "
                  f"sym={sym_err:+.4f} ({expected_str})")

    def _get_diagnostic_positions(self):
        """Return a list of (name, state_input, expected_description) for fixed evaluation."""
        positions = []

        try:
            from games.connect4 import Connect4Game, GameState as C4State
            if not isinstance(self.game, Connect4Game):
                return positions
        except ImportError:
            return positions

        # Position 1: Empty board (X to move) - should be roughly neutral
        s = self.game.new_game()
        positions.append(("empty_board", self.game.state_to_input(s), "expect ~0"))

        # Position 2: X about to win horizontally (X to move)
        # Valid piece counts: X to move → X_count == O_count
        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = -1  # X has 3 in row (cols 0-2)
        board[0][4] = 1     # O pieces scattered (3 total)
        board[0][5] = 1
        board[1][4] = 1
        s = C4State(None, board, player=-1)  # X=3 O=3, X to move
        positions.append(("x_wins_next", self.game.state_to_input(s), "expect > +0.5 (I'm winning)"))

        # Position 3: O about to win horizontally (O to move)
        # Valid piece counts: O to move → X_count == O_count + 1
        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = 1   # O has 3 in row (cols 0-2)
        board[0][4:7] = -1  # X pieces scattered (4 total)
        board[1][4] = -1
        s = C4State(None, board, player=1)  # X=4 O=3, O to move
        positions.append(("o_wins_next", self.game.state_to_input(s), "expect > +0.5 (I'm winning)"))

        # Position 4: The diagonal threat position from the bug report
        board = np.zeros((6, 7), dtype="int")
        board[0] = [0, 1, 1, -1, 0, 1, -1]
        board[1] = [0, -1, 1, -1, 0, 0, 0]
        board[2] = [0, 1, -1, 1, 0, 0, 0]
        board[3] = [0, -1, 0, -1, 0, 0, 0]
        s = C4State(None, board, player=1)  # O to move, X threatens diagonal
        positions.append(("diag_threat", self.game.state_to_input(s), "expect < 0 (I'm losing)"))

        # Position 5: X has strong center control (X to move)
        board = np.zeros((6, 7), dtype="int")
        board[0][3] = -1
        board[1][3] = -1
        board[0][2] = 1
        board[0][4] = 1
        s = C4State(None, board, player=-1)
        positions.append(("x_center", self.game.state_to_input(s), "expect > 0 (I'm slightly winning)"))

        return positions
