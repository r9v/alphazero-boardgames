import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from training.replay_buffer import ReplayBuffer
from training.parallel_self_play import BatchedSelfPlay
from training.training_logger import TrainingLogger
from utils import wdl_to_scalar


def raw_value_to_wdl_class(raw_v):
    """Convert raw values (+1/0/-1) to WDL class indices: +1->0, 0->1, -1->2."""
    return (1 - raw_v).astype(np.int64)



class Trainer:
    def __init__(self, game, net, config=None):
        self.game = game
        self.net = net
        self.config = config or {}
        self.num_simulations = self.config.get("num_simulations", 50)
        self.games_per_iteration = self.config.get("games_per_iteration", 2)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
        self.batch_size = self.config.get("batch_size", 64)
        self.lr = self.config.get("lr", 0.01)
        self.device = self.config.get("device", "cpu")
        self.max_train_steps = self.config.get("max_train_steps", 5000)
        self.target_epochs = self.config.get("target_epochs", 4)
        self.train_ratio = self.config.get("train_ratio", 0)
        self.global_step = 0
        self.global_total_steps = 1
        self.buffer = ReplayBuffer(self.config.get("buffer_size", 100000))

        # Weight decay only on conv/linear weights, not on norm params or biases
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
        self.value_label_smoothing = self.config.get("value_label_smoothing", 0.0)
        self.surprise_weighting = self.config.get("surprise_weighting", True)
        self.surprise_kl_frac = self.config.get("surprise_kl_frac", 0.5)

        self._value_params = [p for n, p in net.named_parameters() if "value" in n]
        self._policy_params = [p for n, p in net.named_parameters() if "policy" in n]

        self.use_amp = (self.device == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        game_name = self.config.get("game_name", "unknown")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = self.config.get("log_dir", f"runs/{game_name}/{timestr}")
        self.writer = SummaryWriter(log_dir)
        self.logger = TrainingLogger(self)

    def train_network(self, n_new_positions=0):
        """Train the network on samples from the replay buffer."""
        samples = [s for s in self.buffer.arr if s is not None]
        if len(samples) < self.batch_size:
            print(f"  Not enough samples ({len(samples)}), skipping training")
            return None

        # Snapshot backbone params for drift measurement
        with torch.no_grad():
            self._pre_train_backbone = torch.cat([
                p.data.flatten() for p in self.net.res_blocks.parameters()
            ]).clone()

        setup = self._init_training_state(samples, n_new_positions)
        samples = setup['train_samples']
        acc = setup['acc']
        cfg = {k: setup[k] for k in ('num_steps', 'effective_vlw', 'effective_epochs',
                                       'early_cutoff', 'late_start', 'lr_min')}

        self.net.train()
        for step in range(cfg['num_steps']):
            lr = self.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.global_step += 1

            if self.surprise_weighting and setup.get('surprise_weights') is not None:
                sw = setup['surprise_weights']
                indices = np.random.choice(len(samples), size=min(self.batch_size, len(samples)),
                                           replace=False, p=sw)
                batch = [samples[i] for i in indices]
            elif (setup['use_stratified'] and len(setup['x_pool']) >= setup['half_batch']
                    and len(setup['o_pool']) >= setup['half_batch']):
                batch = (random.sample(setup['x_pool'], k=setup['half_batch'])
                         + random.sample(setup['o_pool'], k=setup['half_batch']))
            else:
                batch = random.sample(samples, k=min(self.batch_size, len(samples)))

            t0 = time.time()
            states = torch.FloatTensor(np.array([s[0] for s in batch])).to(self.device)
            target_pis = torch.FloatTensor(np.array([s[1] for s in batch])).to(self.device)
            raw_v = np.array([s[2] for s in batch])
            target_vs = torch.LongTensor(raw_value_to_wdl_class(raw_v)).to(self.device)
            acc['data_prep_time'] += time.time() - t0

            t0 = time.time()
            with torch.autocast('cuda', enabled=self.use_amp):
                pred_vs, pred_pi_logits = self.net(states)

                value_loss = F.cross_entropy(pred_vs, target_vs,
                                            label_smoothing=self.value_label_smoothing)

                log_pred_pis = F.log_softmax(pred_pi_logits, dim=1)
                policy_loss = -torch.mean(torch.sum(target_pis * log_pred_pis, dim=1))
                loss = cfg['effective_vlw'] * value_loss + policy_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            self._collect_step_diagnostics(
                step, states, target_vs, target_pis,
                pred_vs, pred_pi_logits, value_loss, policy_loss, acc, cfg)

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            acc['last_batch_grad'] = torch.cat([
                p.grad.flatten() if p.grad is not None
                else torch.zeros(p.numel(), device=self.device)
                for p in self.net.parameters()
            ]).clone()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            acc['gradient_time'] += time.time() - t0

            # Accumulate losses
            acc['total_loss'] += loss.item()
            acc['total_value_loss'] += value_loss.item()
            acc['total_policy_loss'] += policy_loss.item()
            acc['num_batches'] += 1

        # Aggregate results
        return self._aggregate_training_results(
            acc, samples, setup['val_samples'], cfg, setup)

    def _init_training_state(self, samples, n_new_positions):
        """Prepare train/val split, tracking accumulators, and LR schedule params."""
        # Split into train/val for overfitting detection (90/10)
        random.shuffle(samples)
        val_size = max(len(samples) // 10, self.batch_size)
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]

        # Stratified sampling pools: split by player to ensure 50/50 X/O batches
        # X moves when piece counts are equal (ch0==ch1), O when unequal
        x_pool = [s for s in train_samples if s[0][0].sum() == s[0][1].sum()]
        o_pool = [s for s in train_samples if s[0][0].sum() != s[0][1].sum()]

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

        # Policy surprise weights: half uniform, half proportional to KL divergence
        surprise_weights = None
        if self.surprise_weighting:
            kls = np.array([
                s[3].get('policy_surprise', 0.0) if isinstance(s[3], dict) else 0.0
                for s in train_samples
            ], dtype=np.float64)
            mean_kl = kls.mean()
            if mean_kl > 1e-8:
                # Blend: (1-kl_frac)*uniform + kl_frac*KL-proportional
                kl_frac = self.surprise_kl_frac
                uniform = np.ones(len(kls)) / len(kls)
                kl_weights = kls / kls.sum()
                surprise_weights = (1.0 - kl_frac) * uniform + kl_frac * kl_weights
                surprise_weights /= surprise_weights.sum()  # normalize
                acc['mean_policy_surprise'] = float(mean_kl)
                acc['max_policy_surprise'] = float(kls.max())

        return {
            'train_samples': train_samples, 'val_samples': val_samples,
            'x_pool': x_pool, 'o_pool': o_pool,
            'use_stratified': len(x_pool) > 0 and len(o_pool) > 0,
            'half_batch': self.batch_size // 2,
            'num_steps': num_steps, 'effective_vlw': effective_vlw,
            'effective_epochs': effective_epochs,
            'early_cutoff': early_cutoff, 'late_start': late_start,
            'lr_min': lr_min, 'n_samples': n_samples, 'fill_ratio': fill_ratio,
            'surprise_weights': surprise_weights,
            'acc': acc,
        }

    def _collect_step_diagnostics(self, step, states, target_vs, target_pis,
                                   pred_vs, pred_pi_logits, value_loss, policy_loss,
                                   acc, cfg):
        """Sample diagnostics at various intervals during training. Mutates acc."""
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
                    acc['x_pred_sum'] += _scalar_v[is_x].mean().item()
                    acc['x_count'] += 1
                if is_o.any():
                    acc['o_vloss_sum'] += per_sample_vloss[is_o].mean().item()
                    acc['o_target_sum'] += scalar_target[is_o].mean().item()
                    acc['o_pred_sum'] += _scalar_v[is_o].mean().item()
                    acc['o_count'] += 1

                # (#6) Value loss by game phase
                total_pieces = my_counts + opp_counts
                for phase, lo, hi in [('early', 0, 8), ('mid', 9, 20), ('late', 21, 999)]:
                    mask = (total_pieces >= lo) & (total_pieces <= hi)
                    if mask.any():
                        acc['phase_vloss_sums'][phase] += per_sample_vloss[mask].mean().item()
                        acc['phase_counts'][phase] += 1

        # (F) Gradient stats every 50 steps
        if step % 50 == 0:
            with torch.no_grad():
                st = 1 - target_vs.float()
                error_scalar = _scalar_v - st
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
                        'pred_mean': _scalar_v.mean().item(),
                        'target_mean': st.mean().item(),
                        'error_mean': error_scalar.mean().item(),
                        'fc1_per_neuron_gnorm': fc1_per_neuron_gnorm,
                        'vconv_grad_norm': vconv_g.norm().item() if vconv_g is not None else 0.0,
                        'vbn_gamma_grad_norm': vbn_g.norm().item() if vbn_g is not None else 0.0,
                    })

        # Sample gradient norms every 100 steps (uses precomputed param groups)
        if step % 100 == 0:
            v_norm = sum(p.grad.norm().item() ** 2
                         for p in self._value_params if p.grad is not None) ** 0.5
            p_norm = sum(p.grad.norm().item() ** 2
                         for p in self._policy_params if p.grad is not None) ** 0.5
            acc['value_grad_norms'].append(v_norm)
            acc['policy_grad_norms'].append(p_norm)

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
            acc['sub_iter_log'].append({
                'step': step,
                'vloss': value_loss.item(),
                'ploss': policy_loss.item(),
                'v_grad': acc['value_grad_norms'][-1] if acc['value_grad_norms'] else 0,
                'p_grad': acc['policy_grad_norms'][-1] if acc['policy_grad_norms'] else 0,
                'mean_conf': _scalar_v.abs().mean().item(),
                'mean_v': _scalar_v.mean().item(),
            })

        # Track early vs late loss
        if step < early_cutoff:
            acc['early_vloss'] += value_loss.item()
            acc['early_ploss'] += policy_loss.item()
        if step >= late_start:
            acc['late_vloss'] += value_loss.item()
            acc['late_ploss'] += policy_loss.item()

        # Sample predictions from last 10% of steps for distribution analysis
        if step >= late_start:
            acc['all_pred_vs'].append(_scalar_v_np)

            # Value confidence distribution buckets
            _abs_v = np.abs(_scalar_v_np)
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

                # (C) Value confidence calibration (reuse cached _scalar_v)
                scalar_tgt = 1 - target_vs.float()
                confident_mask = _scalar_v.abs() > 0.5
                if confident_mask.any():
                    confident_signs_correct = (
                        _scalar_v[confident_mask].sign() == scalar_tgt[confident_mask].sign()
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

    def _aggregate_training_results(self, acc, samples, val_samples, cfg, setup):
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
                pv, pp_logits = self.net(vs)[:2]
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

        buffer_fill = len(self.buffer)
        buffer_full = buffer_fill >= self.buffer.max_size

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

        self._train_perf = {
            "data_prep_time": acc['data_prep_time'],
            "gradient_time": acc['gradient_time'],
            "num_samples": setup['n_samples'],
            "num_batches": num_batches,
        }
        # SVD rank of last resblock conv2 (capacity indicator)
        svd_info = {}
        try:
            last_rb = self.net.res_blocks[-1]
            w = last_rb.conv2.weight.data.reshape(last_rb.conv2.weight.shape[0], -1)
            sv = torch.linalg.svdvals(w)
            sv_norm = sv / sv[0]
            n_filters = w.shape[0]
            svd_info = {
                'rank90': int((sv_norm > 0.1).sum().item()),
                'rank99': int((sv_norm > 0.01).sum().item()),
                'n_filters': n_filters,
            }
        except Exception:
            pass

        # Backbone drift (cosine similarity pre vs post training)
        drift_cos = None
        if hasattr(self, '_pre_train_backbone'):
            try:
                post_params = torch.cat([p.data.flatten() for p in self.net.res_blocks.parameters()])
                cos = F.cosine_similarity(self._pre_train_backbone.unsqueeze(0),
                                          post_params.unsqueeze(0)).item()
                drift_cos = cos
            except Exception:
                pass

        # Game phase value loss
        gp = {}
        for phase in ('early', 'mid', 'late'):
            c = acc['phase_counts'][phase]
            if c > 0:
                gp[phase] = acc['phase_vloss_sums'][phase] / c

        # Core metrics
        self._train_diag = {
            "avg_value_loss": avg_value_loss,
            "val_vloss": val_vloss,
            "buffer_fill": buffer_fill, "buffer_capacity": self.buffer.max_size,
            "pred_v_std": pred_v_std,
            "policy_top1_acc": policy_top1_acc,
            "rb_grad_norms": avg_rb_grad_norms,
            "svd": svd_info,
            "drift_cos": drift_cos,
            "game_phase_vloss": gp,
        }
        return avg_loss, avg_value_loss, avg_policy_loss

    _SELF_PLAY_KEYS = [
        'selects_per_round', 'vl_value', 'temp_threshold', 'c_puct',
        'dirichlet_alpha', 'tree_reuse', 'random_opening_moves',
        'random_opening_fraction', 'contempt_n',
    ]

    def _self_play(self, iteration):
        """Run self-play games in parallel with batched evaluation."""
        kwargs = {k: self.config[k] for k in self._SELF_PLAY_KEYS if k in self.config}
        self._batched = BatchedSelfPlay(
            self.game, self.net, self.games_per_iteration,
            self.num_simulations, **kwargs)
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

            augmented = []
            for ex in all_examples:
                aux_maps = ex[3] if len(ex) > 3 and isinstance(ex[3], dict) else {}
                syms = self.game.get_symmetries(ex[0], ex[1])
                for sym_tuple in syms:
                    sym_input, sym_policy = sym_tuple[0], sym_tuple[1]
                    entry = [sym_input, sym_policy, ex[2], aux_maps]
                    augmented.append(entry)
            n_new_positions = len(augmented)
            self.buffer._current_iter = iteration
            self.buffer.insert_batch(augmented)

            wins_p1 = results.count(-1)
            wins_p2 = results.count(1)
            draws = results.count(0)
            avg_length = np.mean(game_lengths)
            min_length = int(np.min(game_lengths))
            max_length = int(np.max(game_lengths))
            p1_win_pct = wins_p1 / max(len(results), 1)

            t0 = time.time()
            train_result = self.train_network(n_new_positions=n_new_positions)
            train_time = time.time() - t0

            iter_time = time.time() - iter_t0

            self.logger.log_iteration(iteration, num_iterations, {
                'train_result': train_result,
                'wins_p1': wins_p1, 'wins_p2': wins_p2, 'draws': draws,
                'avg_length': avg_length, 'min_length': min_length,
                'max_length': max_length, 'p1_win_pct': p1_win_pct,
                'self_play_time': self_play_time, 'train_time': train_time,
                'iter_time': iter_time,
            })

            # Save every 5 iterations + always on the last one
            # Also save iteration 0 if no checkpoint exists (quick sanity check)
            no_checkpoint = not os.path.exists(os.path.join(self.checkpoint_dir, "latest.txt"))
            if (iteration + 1) % 5 == 0 or iteration == num_iterations - 1 or (iteration == 0 and no_checkpoint):
                self.net.save(self.checkpoint_dir, iteration=iteration, num_iterations=num_iterations)

        self.logger.close()

