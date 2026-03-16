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
from network.alphazero_net import ws_conv2d


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
        self.ownership_loss_weight = self.config.get("ownership_loss_weight", 0.0)
        self.symmetry_loss_weight = self.config.get("symmetry_loss_weight", 0.0)

        game_name = self.config.get("game_name", "unknown")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_dir = self.config.get("log_dir", f"runs/{game_name}/{timestr}")
        self.writer = SummaryWriter(log_dir)

    def train_network(self):
        """Train the network on samples from the replay buffer."""
        samples = [s for s in self.buffer.arr if s is not None]
        if len(samples) < self.batch_size:
            print(f"  Not enough samples ({len(samples)}), skipping training")
            return None

        # === DIAGNOSTIC: Value target distribution ===
        all_values = np.array([s[2] for s in samples])
        val_mean = all_values.mean()
        val_std = all_values.std()
        frac_pos = (all_values > 0).mean()   # O wins
        frac_neg = (all_values < 0).mean()   # X wins
        frac_draw = (all_values == 0).mean()  # draws
        # Value target distribution histogram (5 bins)
        val_hist = [
            float(((all_values >= -1.0) & (all_values < -0.5)).mean()),  # strong X
            float(((all_values >= -0.5) & (all_values < 0.0)).mean()),   # mild X
            float((all_values == 0.0).mean()),                            # draw
            float(((all_values > 0.0) & (all_values <= 0.5)).mean()),    # mild O
            float(((all_values > 0.5) & (all_values <= 1.0)).mean()),    # strong O
        ]

        # === DIAGNOSTIC: Per-player target bias (BUFFER_BIAS + TARGET_BY_PLAYER) ===
        # Use channel 2 (player indicator plane) to identify whose turn it is
        n_x_buf, n_o_buf = 0, 0
        mean_tgt_x, mean_tgt_o = 0.0, 0.0
        frac_pos_x, frac_pos_o = 0.0, 0.0
        try:
            player_planes = np.array([s[0][2, 0, 0] for s in samples])
            is_x_buf = player_planes < 0  # player=-1 → X to move
            is_o_buf = player_planes > 0  # player=+1 → O to move
            n_x_buf = int(is_x_buf.sum())
            n_o_buf = int(is_o_buf.sum())
            if n_x_buf > 0:
                mean_tgt_x = float(all_values[is_x_buf].mean())
                frac_pos_x = float((all_values[is_x_buf] > 0).mean())
            if n_o_buf > 0:
                mean_tgt_o = float(all_values[is_o_buf].mean())
                frac_pos_o = float((all_values[is_o_buf] > 0).mean())
        except Exception:
            pass

        # === DIAGNOSTIC: 3-in-a-row target bias ===
        # Scan buffer for positions with 3+ consecutive pieces on ch0 vs ch1
        three_r_diag = {}
        try:
            all_inputs = np.array([s[0] for s in samples])  # (N, 2, H, W)
            for ch, ch_name in [(0, 'mine'), (1, 'opp')]:
                ch_data = all_inputs[:, ch]  # (N, H, W)
                # Horizontal 3-in-a-row
                h3 = ch_data[:, :, :-2] * ch_data[:, :, 1:-1] * ch_data[:, :, 2:]
                has_h3 = h3.any(axis=(1, 2))
                # Vertical 3-in-a-row
                v3 = ch_data[:, :-2, :] * ch_data[:, 1:-1, :] * ch_data[:, 2:, :]
                has_v3 = v3.any(axis=(1, 2))
                mask = has_h3 | has_v3
                if ch == 0:
                    _ch0_3r_mask = mask
                count = int(mask.sum())
                mean_target = float(all_values[mask].mean()) if count > 0 else 0.0
                three_r_diag[ch_name] = {
                    'count': count, 'mean_target': mean_target,
                    'frac': count / len(samples),
                }
            # Sign distribution + board dump for ch0 3-in-a-row examples
            if _ch0_3r_mask is not None and _ch0_3r_mask.any():
                _tgts = all_values[_ch0_3r_mask]
                _n_pos = int((_tgts > 0).sum())
                _n_neg = int((_tgts < 0).sum())
                _n_zero = int((_tgts == 0).sum())
                print(f"  Diag[SIGN]: ch0_3row: +:{_n_pos} -:{_n_neg} 0:{_n_zero} "
                      f"(n={len(_tgts)}, mean={_tgts.mean():+.4f})")
                _idx = np.where(_ch0_3r_mask)[0]
                _pos_i = _idx[_tgts > 0][:1] if _n_pos > 0 else np.array([], dtype=int)
                _neg_i = _idx[_tgts < 0][:1] if _n_neg > 0 else np.array([], dtype=int)
                _mid_i = _idx[len(_idx)//2:len(_idx)//2+1]
                _dump = list(dict.fromkeys(
                    [int(x) for x in np.concatenate([_pos_i, _neg_i, _mid_i])]))[:3]
                for _di in _dump:
                    _inp = all_inputs[_di]
                    _t = all_values[_di]
                    _rows = []
                    for _r in range(_inp.shape[1]):
                        _row = ""
                        for _c in range(_inp.shape[2]):
                            _row += "M" if _inp[0,_r,_c] > 0 else ("O" if _inp[1,_r,_c] > 0 else ".")
                        _rows.append(_row)
                    print(f"    [{_di}] t={_t:+.1f} {'/'.join(_rows)}")
        except Exception:
            pass

        # Split into train/val for overfitting detection (90/10)
        random.shuffle(samples)
        val_size = max(len(samples) // 10, self.batch_size)
        val_samples = samples[:val_size]
        train_samples = samples[val_size:]
        samples = train_samples  # train on the 90%

        self.net.train()
        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        total_ownership_loss = 0
        total_symmetry_loss = 0
        num_batches = 0
        data_prep_time = 0.0
        gradient_time = 0.0

        # Track early vs late loss (overfitting within iteration)
        early_vloss = 0.0
        early_ploss = 0.0
        late_vloss = 0.0
        late_ploss = 0.0

        # Track value prediction stats across all batches
        all_pred_vs = []

        # Track gradient norms for value vs policy head
        value_grad_norms = []
        policy_grad_norms = []

        # (A) Per-player value loss breakdown
        x_vloss_sum = 0.0
        o_vloss_sum = 0.0
        x_count = 0
        o_count = 0
        x_target_sum = 0.0
        o_target_sum = 0.0
        x_pred_sum = 0.0
        o_pred_sum = 0.0

        # (F) Gradient direction tracking: does gradient push predictions toward targets?
        grad_correct_count = 0
        grad_total_count = 0

        # (P) Policy quality metrics (sampled in last 10% of steps)
        all_policy_entropy = []
        top1_correct_sum = 0
        top3_correct_sum = 0
        policy_acc_count = 0

        # (C) Value confidence calibration (sampled in last 10% of steps)
        confident_correct_sum = 0
        confident_total = 0

        # (RB) Per-residual-block gradient norms (sampled every 100 steps)
        all_rb_grad_norms = {}

        # Sub-iteration logging (every 100 steps) for intra-iteration dynamics
        sub_iter_log = []

        # Value confidence distribution (from last 10% of steps)
        conf_buckets = {'very_low': 0, 'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        conf_total = 0

        # Intra-iteration value trajectory on FixedEval positions (every ~300 steps)
        fixed_eval_trajectory = []  # list of {step, pos_name: value, ...}

        # Dynamic training steps: target N effective epochs, capped by max_train_steps
        # Scale epochs with buffer fill to prevent memorization when buffer is small
        n_samples = len(samples)
        buffer_capacity = self.buffer.max_size
        fill_ratio = min(n_samples / max(buffer_capacity, 1), 1.0)
        # Linearly scale from 1 epoch (empty) to target_epochs (full)
        scaled_epochs = 1.0 + (self.target_epochs - 1.0) * fill_ratio
        target_steps = int(scaled_epochs * (n_samples // self.batch_size))
        # Ramp vlw from 1.0 to target as buffer fills
        effective_vlw = 1.0 + (self.value_loss_weight - 1.0) * fill_ratio
        num_steps = max(1, min(self.max_train_steps, target_steps))
        effective_epochs = (num_steps * self.batch_size) / n_samples
        early_cutoff = max(num_steps // 10, 1)
        late_start = num_steps - early_cutoff

        # Cosine LR schedule: lr decays from initial to 10% over training steps
        lr_min = self.lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr  # reset to initial at start of each iteration

        # (#2) Pre-training value loss: measure before training to compute delta
        pre_train_vloss = None
        try:
            self.net.eval()
            with torch.no_grad():
                pt_batch = random.choices(samples, k=min(512, len(samples)))
                pt_states = torch.FloatTensor(np.array([s[0] for s in pt_batch])).to(self.device)
                pt_raw_v = np.array([s[2] for s in pt_batch])
                pt_targets_v = torch.LongTensor((1 - pt_raw_v).astype(np.int64)).to(self.device)
                pt_pred_v = self.net(pt_states)[0]
                pre_train_vloss = float(F.cross_entropy(pt_pred_v, pt_targets_v).item())
            self.net.train()
        except Exception:
            pass

        # === DIAGNOSTIC: Per-player predictions before training (PRED_BY_PLAYER) ===
        _pbias_data = None
        try:
            self.net.eval()
            with torch.no_grad():
                pb_batch = random.choices(samples, k=min(512, len(samples)))
                pb_states = torch.FloatTensor(np.array([s[0] for s in pb_batch])).to(self.device)
                pb_targets = np.array([s[2] for s in pb_batch])
                pb_players = np.array([s[0][2, 0, 0] for s in pb_batch])
                pb_out = self.net(pb_states)
                pb_wdl = F.softmax(pb_out[0], dim=1)
                pb_v = (pb_wdl[:, 0] - pb_wdl[:, 2]).cpu().numpy()
                pb_is_x = pb_players < 0
                pb_is_o = pb_players > 0
                _pre_x_pred = float(pb_v[pb_is_x].mean()) if pb_is_x.any() else 0.0
                _pre_o_pred = float(pb_v[pb_is_o].mean()) if pb_is_o.any() else 0.0
                _pre_x_acc = float(((pb_v[pb_is_x] > 0) == (pb_targets[pb_is_x] > 0)).mean()) if pb_is_x.any() else 0.0
                _pre_o_acc = float(((pb_v[pb_is_o] > 0) == (pb_targets[pb_is_o] > 0)).mean()) if pb_is_o.any() else 0.0
                _pbias_data = {
                    'batch': pb_batch, 'targets': pb_targets,
                    'is_x': pb_is_x, 'is_o': pb_is_o,
                    'pre_x_pred': _pre_x_pred, 'pre_o_pred': _pre_o_pred,
                    'pre_x_acc': _pre_x_acc, 'pre_o_acc': _pre_o_acc,
                }
            self.net.train()
        except Exception:
            pass

        # (#6) Value loss by game phase
        phase_vloss_sums = {'early': 0.0, 'mid': 0.0, 'late': 0.0}
        phase_counts = {'early': 0, 'mid': 0, 'late': 0}

        # (#8) Policy loss on value-critical positions
        ploss_decisive_sum = 0.0
        ploss_ambiguous_sum = 0.0
        decisive_count = 0
        ambiguous_count = 0

        for step in range(num_steps):
            # Cosine annealing: lr goes from self.lr -> lr_min
            lr = lr_min + 0.5 * (self.lr - lr_min) * (1 + math.cos(math.pi * step / num_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            batch = random.choices(samples, k=self.batch_size)

            t0 = time.time()
            states = torch.FloatTensor(np.array([s[0] for s in batch])).to(self.device)
            target_pis = torch.FloatTensor(np.array([s[1] for s in batch])).to(self.device)
            raw_v = np.array([s[2] for s in batch])
            target_vs = torch.LongTensor((1 - raw_v).astype(np.int64)).to(self.device)  # WDL class: +1→0, 0→1, -1→2
            has_own = len(batch[0]) > 3 and self.ownership_loss_weight > 0
            if has_own:
                target_own = torch.FloatTensor(np.array([s[3] for s in batch])).to(self.device)
            data_prep_time += time.time() - t0

            t0 = time.time()
            net_out = self.net(states)
            pred_vs, pred_pi_logits = net_out[0], net_out[1]
            pred_own = net_out[2] if len(net_out) > 2 else None

            value_loss = F.cross_entropy(pred_vs, target_vs)
            log_pred_pis = F.log_softmax(pred_pi_logits, dim=1)
            policy_loss = -torch.mean(torch.sum(target_pis * log_pred_pis, dim=1))

            loss = effective_vlw * value_loss + policy_loss

            if has_own and pred_own is not None:
                own_loss = F.mse_loss(pred_own, target_own)
                loss = loss + self.ownership_loss_weight * own_loss
                total_ownership_loss += own_loss.item()

            # === SYMMETRY LOSS: zero-sum constraint V(state) + V(swap(state)) = 0 ===
            if self.symmetry_loss_weight > 0:
                # Player-swap: ch0↔ch1, negate ch2
                states_swap = states.clone()
                states_swap[:, 0], states_swap[:, 1] = states[:, 1].clone(), states[:, 0].clone()
                states_swap[:, 2] = -states[:, 2]
                # Forward pass on swapped states (only need value head)
                swap_out = self.net(states_swap)
                swap_vs = swap_out[0]  # WDL logits
                # Scalar values: P(W) - P(L)
                probs_orig = F.softmax(pred_vs, dim=1)
                probs_swap = F.softmax(swap_vs, dim=1)
                val_orig = probs_orig[:, 0] - probs_orig[:, 2]
                val_swap = probs_swap[:, 0] - probs_swap[:, 2]
                # Detach original: supervised loss governs orig, sym loss governs swap
                sym_loss = ((val_orig.detach() + val_swap) ** 2).mean()
                loss = loss + self.symmetry_loss_weight * sym_loss
                total_symmetry_loss += sym_loss.item()

            self.optimizer.zero_grad()
            loss.backward()

            # (A) Per-player loss breakdown every 10 steps
            if step % 10 == 0:
                with torch.no_grad():
                    # Infer player from piece counts: equal pieces = X to move
                    # Channel 0 = my pieces, channel 1 = opp pieces
                    my_counts = states[:, 0].sum(dim=(1, 2))
                    opp_counts = states[:, 1].sum(dim=(1, 2))
                    is_x = (my_counts == opp_counts)  # X has equal pieces
                    is_o = ~is_x

                    per_sample_vloss = F.cross_entropy(pred_vs, target_vs, reduction='none')
                    # Scalar values for diagnostics: P(win) - P(loss)
                    wdl_probs = F.softmax(pred_vs, dim=1)
                    scalar_v = wdl_probs[:, 0] - wdl_probs[:, 2]  # [B]
                    scalar_target = (1 - target_vs.float())  # class→scalar: 0→+1, 1→0, 2→-1
                    if is_x.any():
                        x_vloss_sum += per_sample_vloss[is_x].mean().item()
                        x_target_sum += scalar_target[is_x].mean().item()
                        x_pred_sum += scalar_v[is_x].mean().item()
                        x_count += 1
                    if is_o.any():
                        o_vloss_sum += per_sample_vloss[is_o].mean().item()
                        o_target_sum += scalar_target[is_o].mean().item()
                        o_pred_sum += scalar_v[is_o].mean().item()
                        o_count += 1

                    # (#6) Value loss by game phase (infer move count from pieces)
                    total_pieces = my_counts + opp_counts  # [batch]
                    for phase, lo, hi in [('early', 0, 8), ('mid', 9, 20), ('late', 21, 999)]:
                        mask = (total_pieces >= lo) & (total_pieces <= hi)
                        if mask.any():
                            phase_vloss_sums[phase] += per_sample_vloss[mask].mean().item()
                            phase_counts[phase] += 1

            # (F) Gradient direction check every 50 steps
            if step % 50 == 0:
                with torch.no_grad():
                    # For CE loss, gradient always pushes in correct direction
                    # (increases probability of correct class)
                    grad_correct_count += 1
                    grad_total_count += 1
                    # Scalar value for tracking
                    wdl_p = F.softmax(pred_vs, dim=1)
                    sv = wdl_p[:, 0] - wdl_p[:, 2]  # P(W) - P(L)
                    st = 1 - target_vs.float()  # class→scalar
                    error_scalar = sv - st
                    # Log value head gradient statistics
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
                value_grad_norms.append(v_norm ** 0.5)
                policy_grad_norms.append(p_norm ** 0.5)

                # (RB) Per-residual-block gradient norms
                for i, block in enumerate(self.net.res_blocks):
                    rb_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in block.parameters() if p.grad is not None
                    ) ** 0.5
                    all_rb_grad_norms.setdefault(i, []).append(rb_norm)

                    # Effective LR: grad_norm / weight_norm for conv2
                    # With BN, this is the actual step size relative to scale
                    c2w = block.conv2.weight
                    if c2w.grad is not None:
                        c2_eff_lr = c2w.grad.norm().item() / max(c2w.data.norm().item(), 1e-8)
                        all_rb_grad_norms.setdefault(f'{i}_eff_lr', []).append(c2_eff_lr)

                # Sub-iteration logging: capture intra-iteration dynamics
                with torch.no_grad():
                    _wdl_sub = F.softmax(pred_vs.detach(), dim=1)
                    _sv_sub = (_wdl_sub[:, 0] - _wdl_sub[:, 2])
                    _mean_conf = _sv_sub.abs().mean().item()
                    _mean_v = _sv_sub.mean().item()
                sub_iter_log.append({
                    'step': step,
                    'vloss': value_loss.item(),
                    'ploss': policy_loss.item(),
                    'v_grad': value_grad_norms[-1] if value_grad_norms else 0,
                    'p_grad': policy_grad_norms[-1] if policy_grad_norms else 0,
                    'mean_conf': _mean_conf,
                    'mean_v': _mean_v,
                })

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            self.optimizer.step()
            gradient_time += time.time() - t0

            # Intra-iteration value trajectory on FixedEval positions (every 300 steps + first + last)
            _fe_inputs = getattr(self, '_fixed_eval_inputs', None)
            _fe_names = getattr(self, '_fixed_eval_names', None)
            if _fe_inputs is not None and _fe_names and (step % 300 == 0 or step == num_steps - 1):
                self.net.eval()
                with torch.no_grad():
                    _fe_v, _fe_p = self.net(_fe_inputs)[:2]
                    _fe_probs = F.softmax(_fe_v, dim=1)
                    _fe_vals = (_fe_probs[:, 0] - _fe_probs[:, 2]).cpu().numpy()
                self.net.train()
                _fe_entry = {'step': step}
                for _fi, _fn in enumerate(_fe_names):
                    _fe_entry[_fn] = float(_fe_vals[_fi])
                fixed_eval_trajectory.append(_fe_entry)

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1

            # Track early vs late loss
            if step < early_cutoff:
                early_vloss += value_loss.item()
                early_ploss += policy_loss.item()
            if step >= late_start:
                late_vloss += value_loss.item()
                late_ploss += policy_loss.item()

            # Sample predictions from last 10% of steps for distribution analysis
            if step >= late_start:
                with torch.no_grad():
                    # Convert WDL logits to scalar values for distribution analysis
                    wdl_p_late = F.softmax(pred_vs.detach(), dim=1)
                    scalar_v_late = (wdl_p_late[:, 0] - wdl_p_late[:, 2]).cpu().numpy()
                all_pred_vs.append(scalar_v_late)

                # Value confidence distribution buckets
                _abs_v = np.abs(scalar_v_late)
                _n = len(_abs_v)
                conf_total += _n
                conf_buckets['very_low'] += int((_abs_v < 0.1).sum())
                conf_buckets['low'] += int(((_abs_v >= 0.1) & (_abs_v < 0.3)).sum())
                conf_buckets['medium'] += int(((_abs_v >= 0.3) & (_abs_v < 0.6)).sum())
                conf_buckets['high'] += int(((_abs_v >= 0.6) & (_abs_v < 0.9)).sum())
                conf_buckets['very_high'] += int((_abs_v >= 0.9).sum())

                # (P) Policy quality metrics
                with torch.no_grad():
                    # Policy entropy: H = -sum(p * log(p))
                    pred_pis = F.softmax(pred_pi_logits.detach(), dim=1)
                    log_pi = F.log_softmax(pred_pi_logits.detach(), dim=1)
                    batch_entropy = -(pred_pis * log_pi).sum(dim=1).mean().item()
                    all_policy_entropy.append(batch_entropy)

                    # Policy top-1 accuracy: does argmax match?
                    pred_top = pred_pi_logits.argmax(dim=1)
                    target_top = target_pis.argmax(dim=1)
                    top1_correct_sum += (pred_top == target_top).float().sum().item()

                    # Policy top-3 accuracy: is MCTS best move in network's top 3?
                    pred_top3 = pred_pi_logits.topk(3, dim=1).indices
                    target_argmax = target_pis.argmax(dim=1).unsqueeze(1)
                    top3_correct_sum += (pred_top3 == target_argmax).any(dim=1).float().sum().item()

                    policy_acc_count += pred_pis.shape[0]

                    # (C) Value confidence calibration (using scalar value from WDL)
                    sv_conf = wdl_p_late[:, 0] - wdl_p_late[:, 2]
                    scalar_tgt = 1 - target_vs.float()  # class→scalar: 0→+1, 1→0, 2→-1
                    confident_mask = sv_conf.abs() > 0.5
                    if confident_mask.any():
                        confident_signs_correct = (
                            sv_conf[confident_mask].sign() == scalar_tgt[confident_mask].sign()
                        ).float()
                        confident_correct_sum += confident_signs_correct.sum().item()
                        confident_total += confident_mask.sum().item()

                    # (#8) Policy loss on value-critical vs ambiguous positions
                    # Decisive = win or loss (class 0 or 2), ambiguous = draw (class 1)
                    decisive_mask = (target_vs != 1)
                    ambig_mask = (target_vs == 1)
                    per_sample_ploss = -torch.sum(target_pis * log_pi, dim=1)  # [batch]
                    if decisive_mask.any():
                        ploss_decisive_sum += per_sample_ploss[decisive_mask].mean().item()
                        decisive_count += 1
                    if ambig_mask.any():
                        ploss_ambiguous_sum += per_sample_ploss[ambig_mask].mean().item()
                        ambiguous_count += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_ownership_loss = total_ownership_loss / max(num_batches, 1)
        avg_symmetry_loss = total_symmetry_loss / max(num_batches, 1)
        early_vloss /= max(early_cutoff, 1)
        early_ploss /= max(early_cutoff, 1)
        late_vloss /= max(early_cutoff, 1)
        late_ploss /= max(early_cutoff, 1)

        # Value prediction distribution (from last 10% of training)
        pred_v_all = np.concatenate(all_pred_vs) if all_pred_vs else np.array([0.0])
        pred_v_mean = pred_v_all.mean()
        pred_v_std = pred_v_all.std()
        pred_v_abs_mean = np.abs(pred_v_all).mean()

        # (P) Policy quality metrics (from last 10% of training)
        avg_policy_entropy = float(np.mean(all_policy_entropy)) if all_policy_entropy else 0.0
        policy_top1_acc = top1_correct_sum / max(policy_acc_count, 1)
        policy_top3_acc = top3_correct_sum / max(policy_acc_count, 1)

        # (C) Value confidence calibration
        value_confidence_acc = confident_correct_sum / max(confident_total, 1)
        value_confident_frac = confident_total / max(policy_acc_count, 1)

        # (RB) Per-block gradient norms
        avg_rb_grad_norms = {}
        for i, norms in all_rb_grad_norms.items():
            avg_rb_grad_norms[i] = float(np.mean(norms))

        # Held-out validation loss (overfitting detection)
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
                vt_v = torch.LongTensor((1 - vt_raw).astype(np.int64)).to(self.device)
                val_out = self.net(vs)
                pv, pp_logits = val_out[0], val_out[1]
                val_vloss += F.cross_entropy(pv, vt_v).item()
                val_ploss += -torch.mean(torch.sum(vt_pi * F.log_softmax(pp_logits, dim=1), dim=1)).item()
                val_batches += 1
            if val_batches > 0:
                val_vloss /= val_batches
                val_ploss /= val_batches
        self.net.train()

        # Gradient dominance: what fraction of loss comes from policy
        policy_frac = avg_policy_loss / max(avg_loss, 1e-8)

        # Gradient norm stats
        avg_value_grad = np.mean(value_grad_norms) if value_grad_norms else 0.0
        avg_policy_grad = np.mean(policy_grad_norms) if policy_grad_norms else 0.0

        # Buffer fill stats
        buffer_fill = sum(1 for s in self.buffer.arr if s is not None)
        buffer_full = buffer_fill >= self.buffer.max_size

        # Theoretical value loss floor: entropy of target distribution (irreducible CE)
        val_loss_floor = 0.0
        for frac in [frac_neg, frac_draw, frac_pos]:
            if frac > 0:
                val_loss_floor -= frac * np.log(frac)

        # (A) Per-player averages
        x_vloss_avg = x_vloss_sum / max(x_count, 1)
        o_vloss_avg = o_vloss_sum / max(o_count, 1)
        x_target_avg = x_target_sum / max(x_count, 1)
        o_target_avg = o_target_sum / max(o_count, 1)
        x_pred_avg = x_pred_sum / max(x_count, 1)
        o_pred_avg = o_pred_sum / max(o_count, 1)

        # (F) Gradient stats summary — computed before vh_diag so per-neuron
        # gradient norms are available for dead/alive split
        grad_stats_summary = {}
        if hasattr(self, '_grad_stats') and self._grad_stats:
            gs = self._grad_stats
            # Aggregate per-neuron gradient norms across all sampled steps
            all_per_neuron = np.stack([g['fc1_per_neuron_gnorm'] for g in gs])  # [steps, neurons]
            avg_per_neuron_gnorm = all_per_neuron.mean(axis=0)  # [neurons]

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

        # (V) Value head health: dead neurons, WDL distribution,
        # weight stats, backbone signal
        vh_diag = {}
        try:
            self.net.eval()
            captured = {}
            def hook_vconv(module, input, output):
                captured['backbone_raw'] = input[0]  # backbone output before value conv
                captured['vconv_out'] = output  # [batch, channels, H, W] (pre-BN)
            def hook_vbn(module, input, output):
                captured['vbn_out'] = output    # [batch, channels, H, W] (post-BN, pre-LeakyReLU)
            def hook_pconv(module, input, output):
                captured['pconv_out'] = output  # policy conv output for comparison
            def hook_fc1(module, input, output):
                captured['fc1_in'] = input[0]   # backbone output (flattened)
                captured['fc1_out'] = output     # linear output (pre-activation)
            def hook_fc2(module, input, output):
                captured['fc2_in'] = input[0]    # post-LeakyReLU, post-dropout
                captured['wdl_logits'] = output   # [batch, 3] raw WDL logits
            # Per-residual-block activation hooks (capture input for residual ratio)
            def make_rb_hook(idx):
                def hook(module, input, output):
                    captured[f'rb{idx}_in'] = input[0]
                    captured[f'rb{idx}_out'] = output
                return hook
            rb_hooks = []
            for idx, block in enumerate(self.net.res_blocks):
                rb_hooks.append(block.register_forward_hook(make_rb_hook(idx)))
            # BN2 hooks (pre-norm: bn2 sits between conv1 and conv2)
            # Also hook conv2 to capture raw residual branch output
            def make_rb_bn2_hook(idx):
                def hook(module, input, output):
                    captured[f'rb{idx}_bn2_in'] = input[0]   # conv1 output (bn2 input)
                    captured[f'rb{idx}_bn2_out'] = output     # bn2 normalized output
                return hook
            def make_rb_conv2_hook(idx):
                def hook(module, input, output):
                    captured[f'rb{idx}_conv2_out'] = output   # raw residual branch output
                return hook
            rb_bn2_hooks = []
            for idx, block in enumerate(self.net.res_blocks):
                rb_bn2_hooks.append(block.bn2.register_forward_hook(make_rb_bn2_hook(idx)))
                rb_bn2_hooks.append(block.conv2.register_forward_hook(make_rb_conv2_hook(idx)))

            h0 = self.net.value_conv.register_forward_hook(hook_vconv)
            h0b = self.net.value_bn.register_forward_hook(hook_vbn)
            h0p = self.net.policy_conv.register_forward_hook(hook_pconv)
            h1 = self.net.value_fc1.register_forward_hook(hook_fc1)
            h2 = self.net.value_fc2.register_forward_hook(hook_fc2)
            with torch.no_grad():
                diag_batch = random.choices(samples, k=min(256, len(samples)))
                diag_inp = torch.FloatTensor(
                    np.array([s[0] for s in diag_batch])
                ).to(self.device)
                v_out = self.net(diag_inp)[0]
            h0.remove()
            h0b.remove()
            h0p.remove()
            h1.remove()
            h2.remove()
            for h in rb_hooks:
                h.remove()
            for h in rb_bn2_hooks:
                h.remove()

            backbone_raw_np = captured['backbone_raw'].cpu().numpy()  # [batch, filters, H, W]
            vconv_np = captured['vconv_out'].cpu().numpy()   # [batch, ch, H, W] (pre-BN)
            vbn_np = captured['vbn_out'].cpu().numpy()       # [batch, ch, H, W] (post-BN)
            pconv_np = captured['pconv_out'].cpu().numpy()   # [batch, 2, H, W]
            fc1_np = captured['fc1_out'].cpu().numpy()       # [batch, neurons]
            fc2_in_np = captured['fc2_in'].cpu().numpy()     # [batch, neurons] post-activation
            wdl_logits_np = captured['wdl_logits'].cpu().numpy()  # [batch, 3]
            wdl_probs_diag = F.softmax(captured['wdl_logits'], dim=1).cpu().numpy()  # [batch, 3]
            scalar_v_diag = wdl_probs_diag[:, 0] - wdl_probs_diag[:, 2]  # [batch]
            fc1_in_np = captured['fc1_in'].cpu().numpy()     # [batch, flat_size]

            # WDL health metrics
            wdl_entropy_per_sample = -(wdl_probs_diag * np.log(wdl_probs_diag + 1e-8)).sum(axis=1)
            wdl_confidence = wdl_probs_diag.max(axis=1)
            # WDL accuracy: compare predicted class vs target class
            diag_raw_targets = np.array([s[2] for s in diag_batch])
            diag_target_classes = (1 - diag_raw_targets).astype(np.int64)
            wdl_pred_classes = wdl_logits_np.argmax(axis=1)
            wdl_accuracy = float((wdl_pred_classes == diag_target_classes).mean())

            n_total_neurons = fc1_np.shape[1]
            # With LeakyReLU, "near-dead" = always near zero (abs < 0.01)
            near_dead = (np.abs(fc1_np) < 0.01).all(axis=0)
            n_dead = int(near_dead.sum())

            # --- fc2 weight stats: are weights growing? (key WDL health metric) ---
            fc2_w = self.net.value_fc2.weight.detach().cpu().numpy()  # [3, fc_size]
            fc2_b = self.net.value_fc2.bias.detach().cpu().numpy()    # [3]

            # --- fc1 weight stats ---
            fc1_w = self.net.value_fc1.weight.detach().cpu().numpy()

            # --- Per-neuron activity: mean abs activation ---
            neuron_abs_mean = np.abs(fc2_in_np).mean(axis=0)  # [neurons]
            active_count = int((neuron_abs_mean > 0.1).sum())

            # --- Metric 2: fc1 activation percentiles (creeping death detection) ---
            fc1_abs = np.abs(fc1_np)  # [batch, neurons]
            fc1_abs_flat = fc1_abs.flatten()
            fc1_p10 = float(np.percentile(fc1_abs_flat, 10))
            fc1_p50 = float(np.percentile(fc1_abs_flat, 50))
            fc1_p90 = float(np.percentile(fc1_abs_flat, 90))
            # Also per-neuron: median abs activation per neuron
            fc1_neuron_medians = np.median(fc1_abs, axis=0)  # [neurons]

            # --- Metric 3: Value conv channel utilization ---
            n_vconv_ch = vconv_np.shape[1]
            vconv_ch_abs_mean = []
            vconv_ch_std = []
            vconv_dead_channels = 0
            for ch in range(n_vconv_ch):
                ch_data = vconv_np[:, ch, :, :]  # [batch, H, W]
                ch_abs = float(np.abs(ch_data).mean())
                ch_s = float(ch_data.std())
                vconv_ch_abs_mean.append(ch_abs)
                vconv_ch_std.append(ch_s)
                if ch_abs < 0.01:
                    vconv_dead_channels += 1

            # --- Metric 4: Per-neuron death tracking ---
            dead_neuron_ids = sorted(np.where(near_dead)[0].tolist())
            # Bottom-5 neurons by activation magnitude (most at-risk)
            weakest_5_idx = np.argsort(neuron_abs_mean)[:5].tolist()
            weakest_5_vals = [float(neuron_abs_mean[i]) for i in weakest_5_idx]

            # --- Metric 5: fc2 weight growth rate (delta from previous iteration) ---
            fc2_w_norm_now = float(np.linalg.norm(fc2_w))
            fc1_w_norm_now = float(np.linalg.norm(fc1_w))
            if not hasattr(self, '_prev_fc2_w_norm'):
                self._prev_fc2_w_norm = fc2_w_norm_now
                self._prev_fc1_w_norm = fc1_w_norm_now
            fc2_w_norm_delta = fc2_w_norm_now - self._prev_fc2_w_norm
            fc1_w_norm_delta = fc1_w_norm_now - self._prev_fc1_w_norm
            self._prev_fc2_w_norm = fc2_w_norm_now
            self._prev_fc1_w_norm = fc1_w_norm_now

            # (#3) Initial conv weight norm
            init_conv_w_norm = float(self.net.conv.weight.data.norm().item())
            init_conv_w_abs_mean = float(self.net.conv.weight.data.abs().mean().item())
            if not hasattr(self, '_prev_init_conv_w_norm'):
                self._prev_init_conv_w_norm = init_conv_w_norm
            init_conv_w_norm_delta = init_conv_w_norm - self._prev_init_conv_w_norm
            self._prev_init_conv_w_norm = init_conv_w_norm

            # (#6) Per-conv weight norms inside ResBlocks
            rb_conv_norms = {}
            if not hasattr(self, '_prev_rb_c2_norms'):
                self._prev_rb_c2_norms = {}
            for bi in range(len(self.net.res_blocks)):
                c1_norm = float(self.net.res_blocks[bi].conv1.weight.data.norm().item())
                c2_norm = float(self.net.res_blocks[bi].conv2.weight.data.norm().item())
                c2_delta = c2_norm - self._prev_rb_c2_norms.get(bi, c2_norm)
                self._prev_rb_c2_norms[bi] = c2_norm
                rb_conv_norms[bi] = {"conv1": c1_norm, "conv2": c2_norm, "c2_delta": c2_delta}

            # --- Metric 1: Gradient flow split by alive/dead neurons ---
            grad_dead_mean = 0.0
            grad_alive_mean = 0.0
            gs_data = grad_stats_summary.get('fc1_per_neuron_gnorm')
            if gs_data is not None and len(gs_data) == n_total_neurons:
                dead_mask_arr = near_dead  # boolean [neurons]
                alive_mask_arr = ~dead_mask_arr
                if dead_mask_arr.any():
                    grad_dead_mean = float(gs_data[dead_mask_arr].mean())
                if alive_mask_arr.any():
                    grad_alive_mean = float(gs_data[alive_mask_arr].mean())

            # --- Value conv decay diagnostics ---
            # (a) Backbone raw signal (input to value_conv)
            backbone_raw_abs = float(np.abs(backbone_raw_np).mean())
            backbone_raw_std = float(backbone_raw_np.std())

            # (i) Per-channel backbone magnitude distribution
            bb_ch_abs = np.abs(backbone_raw_np).mean(axis=(0, 2, 3))  # [channels]
            bb_n_channels = len(bb_ch_abs)
            bb_dead_channels = int((bb_ch_abs < 0.01).sum())
            bb_ch_p10 = float(np.percentile(bb_ch_abs, 10))
            bb_ch_p50 = float(np.percentile(bb_ch_abs, 50))
            bb_ch_p90 = float(np.percentile(bb_ch_abs, 90))
            bb_ch_max = float(bb_ch_abs.max())
            # Top-5 and bottom-5 channels by magnitude
            bb_top5_idx = np.argsort(bb_ch_abs)[-5:][::-1].tolist()
            bb_top5_vals = [float(bb_ch_abs[i]) for i in bb_top5_idx]
            bb_bot5_idx = np.argsort(bb_ch_abs)[:5].tolist()
            bb_bot5_vals = [float(bb_ch_abs[i]) for i in bb_bot5_idx]

            # (j) Per-residual-block activation magnitude
            rb_act_stats = {}
            for idx in range(len(self.net.res_blocks)):
                key = f'rb{idx}_out'
                if key in captured:
                    rb_np = captured[key].cpu().numpy()  # [batch, channels, H, W]
                    rb_abs = float(np.abs(rb_np).mean())
                    rb_std_val = float(rb_np.std())
                    rb_ch_abs = np.abs(rb_np).mean(axis=(0, 2, 3))  # [channels]
                    rb_dead = int((rb_ch_abs < 0.01).sum())
                    rb_act_stats[idx] = {
                        "abs_mean": rb_abs,
                        "std": rb_std_val,
                        "dead_channels": rb_dead,
                        "ch_p50": float(np.percentile(rb_ch_abs, 50)),
                    }

            # (#5) Residual contribution ratio: ||residual|| / ||skip||
            rb_residual_ratios = {}
            for idx in range(len(self.net.res_blocks)):
                in_key = f'rb{idx}_in'
                out_key = f'rb{idx}_out'
                if in_key in captured and out_key in captured:
                    rb_in = captured[in_key]  # skip connection (input)
                    rb_out = captured[out_key]  # output = skip + residual
                    residual = rb_out - rb_in
                    skip_norm = float(rb_in.norm().item())
                    res_norm = float(residual.norm().item())
                    rb_residual_ratios[idx] = res_norm / max(skip_norm, 1e-10)

            # (#7) Residual branch output stats (pre-norm: conv2 output goes directly to skip)
            # Track conv2 raw output magnitude — this is what gets added to the skip.
            # Also track BN2 stats (bn2 is between conv1 and conv2 in pre-norm).
            rb_bn2_stats = {}
            for idx in range(len(self.net.res_blocks)):
                conv2_out_key = f'rb{idx}_conv2_out'
                bn2_out_key = f'rb{idx}_bn2_out'
                bn2_in_key = f'rb{idx}_bn2_in'
                stats = {}
                # Conv2 raw output = residual branch output (no BN after it)
                if conv2_out_key in captured:
                    conv2_out = captured[conv2_out_key].cpu().numpy()
                    stats["conv2_raw_var"] = float(conv2_out.var())
                    stats["conv2_raw_abs"] = float(np.abs(conv2_out).mean())
                # BN2 output (feeds into ReLU→Conv2)
                if bn2_out_key in captured:
                    bn2_out = captured[bn2_out_key].cpu().numpy()
                    stats["bn2_out_std"] = float(bn2_out.std())
                    stats["bn2_out_abs"] = float(np.abs(bn2_out).mean())
                # BN2 batch vs running variance
                if bn2_in_key in captured:
                    bn2_in_t = captured[bn2_in_key]  # [B, C, H, W]
                    batch_var = bn2_in_t.var(dim=(0, 2, 3))  # [C]
                    run_var = self.net.res_blocks[idx].bn2.running_var
                    if run_var is not None:
                        ratio = (batch_var / (run_var + 1e-5)).cpu().numpy()
                        stats["bn2_batch_vs_run_var_mean"] = float(ratio.mean())
                        stats["bn2_batch_vs_run_var_std"] = float(ratio.std())
                        stats["bn2_batch_var_mean"] = float(batch_var.mean().item())
                        stats["bn2_run_var_mean"] = float(run_var.mean().item())
                if stats:
                    rb_bn2_stats[idx] = stats

            # (#11) Residual path effective rank
            # SVD of mean residual (output - input) per block.
            # Low rank = residual collapsed to few directions.
            rb_res_rank = {}
            for idx in range(len(self.net.res_blocks)):
                in_key = f'rb{idx}_in'
                out_key = f'rb{idx}_out'
                if in_key in captured and out_key in captured:
                    residual = captured[out_key] - captured[in_key]
                    r_mean = residual.mean(dim=0).flatten(1)  # [C, H*W]
                    svs = torch.linalg.svdvals(r_mean)
                    energy = (svs**2).cumsum(0) / (svs**2).sum()
                    rank90 = int((energy <= 0.9).sum().item()) + 1
                    rank99 = int((energy <= 0.99).sum().item()) + 1
                    rb_res_rank[idx] = {
                        "rank90": rank90, "rank99": rank99,
                        "total": r_mean.shape[0],
                    }

            # (b) Pre-BN signal (value_conv output, before BatchNorm)
            vconv_pre_bn_abs = float(np.abs(vconv_np).mean())
            vconv_pre_bn_std = float(vconv_np.std())

            # (c) Post-BN signal (after BatchNorm, before LeakyReLU)
            vbn_abs = float(np.abs(vbn_np).mean())
            vbn_std = float(vbn_np.std())

            # (d) BN suppression ratio: post-BN / pre-BN magnitude
            bn_ratio = vbn_abs / max(vconv_pre_bn_abs, 1e-10)

            # (e) value_conv weight stats
            vc_w = self.net.value_conv.weight.detach().cpu().numpy()
            vc_w_norm = float(np.linalg.norm(vc_w))
            vc_w_abs_mean = float(np.abs(vc_w).mean())

            # (f) value_bn gamma (weight) and beta (bias) per channel
            vbn_gamma = self.net.value_bn.weight.detach().cpu().numpy()   # [channels]
            vbn_beta = self.net.value_bn.bias.detach().cpu().numpy()      # [channels]
            vbn_running_var = self.net.value_bn.running_var.detach().cpu().numpy()

            # (g) Policy conv signal for comparison
            pconv_abs = float(np.abs(pconv_np).mean())
            pconv_std = float(pconv_np.std())

            # (h) value_conv gradient norm (from training steps)
            vconv_grad_norm = grad_stats_summary.get('vconv_grad_norm', 0.0)

            vh_diag = {
                "dead_neurons": n_dead,
                "total_neurons": n_total_neurons,
                "active_neurons": active_count,
                # WDL metrics (replaces pre_tanh/saturation)
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
                # fc2 weight diagnostics (key WDL health: should stabilize)
                "fc2_w_max": float(fc2_w.max()),
                "fc2_w_min": float(fc2_w.min()),
                "fc2_w_norm": fc2_w_norm_now,
                "fc2_bias_w": float(fc2_b[0]),
                "fc2_bias_d": float(fc2_b[1]),
                "fc2_bias_l": float(fc2_b[2]),
                # fc1 weight diagnostics
                "fc1_w_norm": fc1_w_norm_now,
                # Backbone signal into value head
                "backbone_std": float(fc1_in_np.std()),
                "backbone_abs_mean": float(np.abs(fc1_in_np).mean()),
                # Metric 1: Gradient flow to dead vs alive neurons
                "grad_dead_mean": grad_dead_mean,
                "grad_alive_mean": grad_alive_mean,
                # Metric 2: fc1 activation percentiles
                "fc1_act_p10": fc1_p10,
                "fc1_act_p50": fc1_p50,
                "fc1_act_p90": fc1_p90,
                # Metric 3: Value conv channel utilization
                "vconv_dead_channels": vconv_dead_channels,
                "vconv_n_channels": n_vconv_ch,
                "vconv_ch_abs_mean": vconv_ch_abs_mean,
                "vconv_ch_std": vconv_ch_std,
                # Metric 4: Per-neuron death tracking
                "dead_neuron_ids": dead_neuron_ids,
                "weakest_5_ids": weakest_5_idx,
                "weakest_5_vals": weakest_5_vals,
                # Metric 5: Weight growth rate
                "fc2_w_norm_delta": fc2_w_norm_delta,
                "fc1_w_norm_delta": fc1_w_norm_delta,
                # Value conv decay diagnostics
                "backbone_raw_abs": backbone_raw_abs,
                "backbone_raw_std": backbone_raw_std,
                "vconv_pre_bn_abs": vconv_pre_bn_abs,
                "vconv_pre_bn_std": vconv_pre_bn_std,
                "vbn_post_abs": vbn_abs,
                "vbn_post_std": vbn_std,
                "bn_ratio": bn_ratio,
                "vc_w_norm": vc_w_norm,
                "vc_w_abs_mean": vc_w_abs_mean,
                "vbn_gamma": vbn_gamma.tolist(),
                "vbn_beta": vbn_beta.tolist(),
                "vbn_gamma_mean": float(vbn_gamma.mean()),
                "vbn_gamma_min": float(vbn_gamma.min()),
                "vbn_running_var": vbn_running_var.tolist(),
                "pconv_abs": pconv_abs,
                "pconv_std": pconv_std,
                "vconv_grad_norm": vconv_grad_norm,
                # Backbone per-channel stats
                "bb_n_channels": bb_n_channels,
                "bb_dead_channels": bb_dead_channels,
                "bb_ch_p10": bb_ch_p10,
                "bb_ch_p50": bb_ch_p50,
                "bb_ch_p90": bb_ch_p90,
                "bb_ch_max": bb_ch_max,
                "bb_top5": list(zip(bb_top5_idx, bb_top5_vals)),
                "bb_bot5": list(zip(bb_bot5_idx, bb_bot5_vals)),
                # Per-block activation stats
                "rb_act_stats": rb_act_stats,
                # (#5) Residual contribution ratio
                "rb_residual_ratios": rb_residual_ratios,
                # (#3) Initial conv weight norm
                "init_conv_w_norm": init_conv_w_norm,
                "init_conv_w_abs_mean": init_conv_w_abs_mean,
                "init_conv_w_norm_delta": init_conv_w_norm_delta,
                # (#6) Per-conv weight norms in ResBlocks
                "rb_conv_norms": rb_conv_norms,
                # (#7) BN2 output scale + (#8) pre-BN2 variance
                "rb_bn2_stats": rb_bn2_stats,
                # (#11) Residual path effective rank
                "rb_res_rank": rb_res_rank,
            }
            self.net.train()
        except Exception as e:
            print(f"  [DIAG-DBG] Value head diagnostic block failed: {e}")

        # === Backbone gradient decomposition: value vs policy ===
        # Separate backward passes to see which head drives each backbone channel
        try:
            self.net.eval()  # Don't update BN running stats
            gd_batch = random.choices(samples, k=min(64, len(samples)))
            gd_states = torch.FloatTensor(np.array([s[0] for s in gd_batch])).to(self.device)
            gd_raw_v = np.array([s[2] for s in gd_batch])
            gd_targets_v = torch.LongTensor((1 - gd_raw_v).astype(np.int64)).to(self.device)
            gd_targets_pi = torch.FloatTensor(np.array([s[1] for s in gd_batch])).to(self.device)

            # Hook to capture backbone output tensor (with grad tracking)
            bb_ref = {}
            def hook_bb_capture(module, inp, out):
                bb_ref['x'] = inp[0]
            h_bb = self.net.value_conv.register_forward_hook(hook_bb_capture)
            # Hooks for per-block channel dominance (#9)
            rb_gd_refs = {}
            def make_rb_gd_hook(idx):
                def hook(module, input, output):
                    rb_gd_refs[f'rb{idx}'] = output
                return hook
            rb_gd_hooks = []
            for idx, block in enumerate(self.net.res_blocks):
                rb_gd_hooks.append(block.register_forward_hook(make_rb_gd_hook(idx)))

            self.optimizer.zero_grad()
            gd_pred_v, gd_pred_p_logits = self.net(gd_states)[:2]

            gd_v_loss = F.cross_entropy(gd_pred_v, gd_targets_v)
            gd_p_loss = -torch.mean(torch.sum(gd_targets_pi * F.log_softmax(gd_pred_p_logits, dim=1), dim=1))

            bb_x = bb_ref['x']  # [batch, 256, H, W]
            h_bb.remove()

            # (A) Per-channel gradient from each head on backbone output
            v_grad_bb = torch.autograd.grad(gd_v_loss, bb_x, retain_graph=True)[0]
            p_grad_bb = torch.autograd.grad(gd_p_loss, bb_x, retain_graph=True)[0]

            v_grad_ch = v_grad_bb.abs().mean(dim=(0, 2, 3)).cpu().numpy()  # [channels]
            p_grad_ch = p_grad_bb.abs().mean(dim=(0, 2, 3)).cpu().numpy()  # [channels]

            v_grad_bb_norm = float(v_grad_bb.norm().item())
            p_grad_bb_norm = float(p_grad_bb.norm().item())

            # (#1) Backbone gradient conflict: cosine similarity between v and p gradients
            bb_grad_cosine = float(F.cosine_similarity(
                v_grad_bb.flatten().unsqueeze(0),
                p_grad_bb.flatten().unsqueeze(0)
            ).item())
            # Per-channel cosine: how many channels have conflicting gradient directions?
            v_ch_flat = v_grad_bb.mean(dim=0).flatten(1)  # [channels, H*W]
            p_ch_flat = p_grad_bb.mean(dim=0).flatten(1)  # [channels, H*W]
            ch_cosines = F.cosine_similarity(v_ch_flat, p_ch_flat, dim=1)  # [channels]
            bb_grad_conflict_channels = int((ch_cosines < 0).sum().item())
            bb_grad_aligned_channels = int((ch_cosines > 0.5).sum().item())

            # Per-channel dominance: what fraction of gradient comes from value?
            ch_total = v_grad_ch + p_grad_ch + 1e-10
            ch_value_frac = v_grad_ch / ch_total  # [channels]
            n_value_dom = int((ch_value_frac > 0.5).sum())
            n_policy_dom = int((ch_value_frac <= 0.5).sum())

            # Top value-dominated and policy-dominated channels
            top_v_ch = np.argsort(ch_value_frac)[-5:][::-1].tolist()
            top_p_ch = np.argsort(ch_value_frac)[:5].tolist()
            top_v_frac = [float(ch_value_frac[i]) for i in top_v_ch]
            top_p_frac = [float(ch_value_frac[i]) for i in top_p_ch]

            # (C) Value-Policy weight correlation on backbone channels
            value_w = self.net.value_conv.weight.data  # [vch, 256, 1, 1]
            policy_w = self.net.policy_conv.weight.data  # [pch, 256, 1, 1]
            val_per_ch = value_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()  # [256]
            pol_per_ch = policy_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()  # [256]
            vp_corr = float(np.corrcoef(val_per_ch, pol_per_ch)[0, 1])
            top20_val = np.argsort(val_per_ch)[-20:]
            top20_pol = np.argsort(pol_per_ch)[-20:]
            vp_overlap_20 = len(set(top20_val.tolist()) & set(top20_pol.tolist()))
            bb_act_ch = bb_x.detach().abs().mean(dim=(0, 2, 3)).cpu().numpy()  # [256]
            val_top20_act = float(bb_act_ch[top20_val].mean())
            pol_top20_act = float(bb_act_ch[top20_pol].mean())
            val_health_corr = float(np.corrcoef(val_per_ch, bb_act_ch)[0, 1])
            pol_health_corr = float(np.corrcoef(pol_per_ch, bb_act_ch)[0, 1])

            # (B) Backbone parameter gradient decomposition
            backbone_params = []
            for name, param in self.net.named_parameters():
                if not name.startswith('value') and not name.startswith('policy'):
                    backbone_params.append((name, param))

            bp_list = [p for _, p in backbone_params]
            v_bp_grads = torch.autograd.grad(gd_v_loss, bp_list,
                                             retain_graph=True, allow_unused=True)
            p_bp_grads = torch.autograd.grad(gd_p_loss, bp_list,
                                             retain_graph=True, allow_unused=True)

            v_bp_sq = 0.0
            p_bp_sq = 0.0
            # Per res-block gradient breakdown
            rb_v = {}
            rb_p = {}
            for (name, param), vg, pg in zip(backbone_params, v_bp_grads, p_bp_grads):
                vn = vg.norm().item() if vg is not None else 0.0
                pn = pg.norm().item() if pg is not None else 0.0
                v_bp_sq += vn ** 2
                p_bp_sq += pn ** 2
                # Group by res block number
                if name.startswith('res_blocks.'):
                    bn = name.split('.')[1]  # block number
                    rb_v[bn] = rb_v.get(bn, 0.0) + vn ** 2
                    rb_p[bn] = rb_p.get(bn, 0.0) + pn ** 2

            v_bp_total = v_bp_sq ** 0.5
            p_bp_total = p_bp_sq ** 0.5

            # Per res-block summary
            rb_summary = {}
            for bn in sorted(rb_v.keys()):
                rv = rb_v[bn] ** 0.5
                rp = rb_p.get(bn, 0.0) ** 0.5
                rb_summary[bn] = (rv, rp, rv / max(rp, 1e-10))

            # (#9) Per-block channel dominance: value-dominated channels at each block output
            rb_ch_dominance = {}
            for idx in range(len(self.net.res_blocks)):
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

            # (#10) Value gradient survival ratio through backbone depth
            # How much of value gradient at backbone output survives to each block's parameters
            rb_v_grad_survival = {}
            for bn in sorted(rb_v.keys()):
                rv = rb_v[bn] ** 0.5
                rb_v_grad_survival[bn] = rv / max(v_grad_bb_norm, 1e-10)

            for h in rb_gd_hooks:
                h.remove()

            self.optimizer.zero_grad()
            self.net.train()

            vh_diag.update({
                # Backbone output gradient decomposition
                "bb_v_grad_norm": v_grad_bb_norm,
                "bb_p_grad_norm": p_grad_bb_norm,
                "bb_grad_ratio": v_grad_bb_norm / max(p_grad_bb_norm, 1e-10),
                "bb_n_value_dom": n_value_dom,
                "bb_n_policy_dom": n_policy_dom,
                "bb_top_v_channels": list(zip(top_v_ch, top_v_frac)),
                "bb_top_p_channels": list(zip(top_p_ch, top_p_frac)),
                # Backbone parameter gradient decomposition
                "bb_param_v_grad": v_bp_total,
                "bb_param_p_grad": p_bp_total,
                "bb_param_grad_ratio": v_bp_total / max(p_bp_total, 1e-10),
                "bb_res_block_grads": rb_summary,
                # Value-Policy competition
                "vp_weight_corr": vp_corr,
                "vp_overlap_20": vp_overlap_20,
                "vp_val_top20_act": val_top20_act,
                "vp_pol_top20_act": pol_top20_act,
                "vp_val_health_corr": val_health_corr,
                "vp_pol_health_corr": pol_health_corr,
                # (#1) Backbone gradient conflict
                "bb_grad_cosine_sim": bb_grad_cosine,
                "bb_grad_conflict_channels": bb_grad_conflict_channels,
                "bb_grad_aligned_channels": bb_grad_aligned_channels,
                # (#9) Per-block channel dominance
                "rb_ch_dominance": rb_ch_dominance,
                # (#10) Value gradient survival
                "rb_v_grad_survival": rb_v_grad_survival,
            })
        except Exception as e:
            print(f"  [DIAG-DBG] Backbone gradient decomposition failed: {e}")

        # === SVD rank tracking & policy head internals ===
        try:
            self.net.eval()
            # Backbone: SVD of deepest res block conv2 (most compressed layer)
            rb_idx = len(self.net.res_blocks) - 1
            rb_w = self.net.res_blocks[rb_idx].conv2.weight.data.flatten(1)
            svs_bb = torch.linalg.svdvals(rb_w)
            energy_bb = (svs_bb**2).cumsum(0) / (svs_bb**2).sum()
            bb_rank90 = int((energy_bb <= 0.9).sum().item()) + 1
            bb_rank99 = int((energy_bb <= 0.99).sum().item()) + 1
            bb_near_zero_sv = int((svs_bb < 1e-6).sum().item())
            bb_n = rb_w.shape[0]

            # Backbone BN: dead channel count in deepest bn2
            bn_gamma = self.net.res_blocks[rb_idx].bn2.weight.data
            bn_var = self.net.res_blocks[rb_idx].bn2.running_var
            if bn_var is not None:
                eff_gain = bn_gamma / (bn_var + 1e-5).sqrt()
                bn_dead = int((eff_gain.abs() < 0.1).sum().item())
            else:
                bn_dead = -1

            # Policy FC: SVD effective rank
            pfc_w = self.net.policy_fc.weight.data
            svs_pfc = torch.linalg.svdvals(pfc_w)
            energy_pfc = (svs_pfc**2).cumsum(0) / (svs_pfc**2).sum()
            pfc_max_rank = min(pfc_w.shape[0], pfc_w.shape[1])
            pfc_rank90 = int((energy_pfc <= 0.9).sum().item()) + 1
            pfc_rank99 = int((energy_pfc <= 0.99).sum().item()) + 1

            # Policy conv: per-channel L1 norm
            pc_w = self.net.policy_conv.weight.data.squeeze(-1).squeeze(-1)
            if pc_w.dim() == 1:
                pc_w = pc_w.unsqueeze(0)
            pc_ch_norms = pc_w.abs().sum(dim=1).cpu().tolist()

            # Per-block BN diagnostics (all res blocks)
            all_rb_bn = {}
            for bi in range(len(self.net.res_blocks)):
                rb_bn_gamma = self.net.res_blocks[bi].bn2.weight.data
                rb_bn_var = self.net.res_blocks[bi].bn2.running_var
                if rb_bn_var is not None:
                    rb_eff_gain = rb_bn_gamma / (rb_bn_var + 1e-5).sqrt()
                    rb_bn_dead = int((rb_eff_gain.abs() < 0.1).sum().item())
                    rb_neg_gamma = int((rb_bn_gamma < -0.01).sum().item())
                    rb_eff_gain_np = rb_eff_gain.cpu().numpy()
                    rb_sqrt_var = (rb_bn_var + 1e-5).sqrt()
                    all_rb_bn[bi] = {
                        "dead": rb_bn_dead,
                        "neg_gamma": rb_neg_gamma,
                        "eff_gain_mean": float(np.abs(rb_eff_gain_np).mean()),
                        "eff_gain_max": float(np.abs(rb_eff_gain_np).max()),
                        "gamma_mean": float(rb_bn_gamma.abs().mean().item()),
                        "gamma_std": float(rb_bn_gamma.std().item()),
                        "sqrt_var_mean": float(rb_sqrt_var.mean().item()),
                        "sqrt_var_std": float(rb_sqrt_var.std().item()),
                    }
                # SVD rank for each block's conv2
                rb_conv2_w = self.net.res_blocks[bi].conv2.weight.data.flatten(1)
                rb_svs = torch.linalg.svdvals(rb_conv2_w)
                rb_energy = (rb_svs**2).cumsum(0) / (rb_svs**2).sum()
                rb_rank90 = int((rb_energy <= 0.9).sum().item()) + 1
                if bi not in all_rb_bn:
                    all_rb_bn[bi] = {}
                all_rb_bn[bi]["svd_rank90"] = rb_rank90
                all_rb_bn[bi]["svd_total"] = rb_conv2_w.shape[0]

            # (#4) final_bn gamma tracking (for pre-act networks)
            fbn_eff_gain_mean = fbn_eff_gain_min = fbn_eff_gain_max = 0.0
            fbn_dead = -1
            fbn_gamma_mean = fbn_gamma_std = 0.0
            fbn_sqrt_var_mean = fbn_sqrt_var_std = 0.0
            try:
                fbn = getattr(self.net, 'final_bn', None)
                if fbn is not None:
                    fbn_var = fbn.running_var
                    if fbn_var is not None:
                        fbn_gamma = fbn.weight.data
                        fbn_eff_gain = fbn_gamma / (fbn_var + 1e-5).sqrt()
                        fbn_eff_gain_np = fbn_eff_gain.cpu().numpy()
                        fbn_eff_gain_mean = float(np.abs(fbn_eff_gain_np).mean())
                        fbn_eff_gain_min = float(np.abs(fbn_eff_gain_np).min())
                        fbn_eff_gain_max = float(np.abs(fbn_eff_gain_np).max())
                        fbn_dead = int((fbn_eff_gain.abs() < 0.1).sum().item())
                        fbn_sqrt_var = (fbn_var + 1e-5).sqrt()
                        fbn_gamma_mean = float(fbn_gamma.abs().mean().item())
                        fbn_gamma_std = float(fbn_gamma.std().item())
                        fbn_sqrt_var_mean = float(fbn_sqrt_var.mean().item())
                        fbn_sqrt_var_std = float(fbn_sqrt_var.std().item())
                    else:
                        print("  [FBN-DBG] final_bn.running_var is None")
                else:
                    print("  [FBN-DBG] final_bn not found on net")
            except Exception as e:
                print(f"  [FBN-DBG] Exception in FBN diagnostic: {e}")

            vh_diag.update({
                "svd_bb_rank90": bb_rank90,
                "svd_bb_rank99": bb_rank99,
                "svd_bb_total": bb_n,
                "svd_bb_near_zero": bb_near_zero_sv,
                "bn_dead_deepest": bn_dead,
                "svd_pfc_rank90": pfc_rank90,
                "svd_pfc_rank99": pfc_rank99,
                "svd_pfc_max_rank": pfc_max_rank,
                "pconv_ch_norms": pc_ch_norms,
                "all_rb_bn": all_rb_bn,
                # (#4) final_bn
                "final_bn_eff_gain_mean": fbn_eff_gain_mean,
                "final_bn_eff_gain_min": fbn_eff_gain_min,
                "final_bn_eff_gain_max": fbn_eff_gain_max,
                "final_bn_dead": fbn_dead,
                "final_bn_gamma_mean": fbn_gamma_mean,
                "final_bn_gamma_std": fbn_gamma_std,
                "final_bn_sqrt_var_mean": fbn_sqrt_var_mean,
                "final_bn_sqrt_var_std": fbn_sqrt_var_std,
            })
            self.net.train()
        except Exception as e:
            print(f"  [DIAG-DBG] SVD/rank diagnostic block failed: {e}")

        # === DIAGNOSTIC: Per-player predictions after training (PRED_BY_PLAYER) ===
        _post_x_pred, _post_o_pred = 0.0, 0.0
        _post_x_acc, _post_o_acc = 0.0, 0.0
        if _pbias_data is not None:
            try:
                self.net.eval()
                with torch.no_grad():
                    pb_states_post = torch.FloatTensor(
                        np.array([s[0] for s in _pbias_data['batch']])
                    ).to(self.device)
                    pb_out_post = self.net(pb_states_post)
                    pb_wdl_post = F.softmax(pb_out_post[0], dim=1)
                    pb_v_post = (pb_wdl_post[:, 0] - pb_wdl_post[:, 2]).cpu().numpy()
                    _pb_targets = _pbias_data['targets']
                    _pb_is_x = _pbias_data['is_x']
                    _pb_is_o = _pbias_data['is_o']
                    if _pb_is_x.any():
                        _post_x_pred = float(pb_v_post[_pb_is_x].mean())
                        _post_x_acc = float(((pb_v_post[_pb_is_x] > 0) == (_pb_targets[_pb_is_x] > 0)).mean())
                    if _pb_is_o.any():
                        _post_o_pred = float(pb_v_post[_pb_is_o].mean())
                        _post_o_acc = float(((pb_v_post[_pb_is_o] > 0) == (_pb_targets[_pb_is_o] > 0)).mean())
                self.net.train()
            except Exception:
                pass

        self._train_perf = {
            "data_prep_time": data_prep_time,
            "gradient_time": gradient_time,
            "num_samples": n_samples,
            "num_batches": num_batches,
        }
        self._train_diag = {
            "val_target_mean": val_mean,
            "val_target_std": val_std,
            "frac_pos": frac_pos,
            "frac_neg": frac_neg,
            "frac_draw": frac_draw,
            "effective_epochs": effective_epochs,
            "num_steps": num_steps,
            "early_vloss": early_vloss,
            "early_ploss": early_ploss,
            "late_vloss": late_vloss,
            "late_ploss": late_ploss,
            "val_vloss": val_vloss,
            "val_ploss": val_ploss,
            "buffer_fill": buffer_fill,
            "buffer_capacity": self.buffer.max_size,
            "buffer_full": buffer_full,
            "pred_v_mean": pred_v_mean,
            "pred_v_std": pred_v_std,
            "pred_v_abs_mean": pred_v_abs_mean,
            "policy_grad_frac": policy_frac,
            "val_loss_floor": val_loss_floor,
            "avg_value_grad_norm": avg_value_grad,
            "avg_policy_grad_norm": avg_policy_grad,
            # (A) Per-player breakdown
            "x_vloss": x_vloss_avg,
            "o_vloss": o_vloss_avg,
            "x_target_mean": x_target_avg,
            "o_target_mean": o_target_avg,
            "x_pred_mean": x_pred_avg,
            "o_pred_mean": o_pred_avg,
            # (F) Gradient direction stats
            "grad_stats": grad_stats_summary,
            # (V) Value head health
            "vh_diag": vh_diag,
            "effective_vlw": effective_vlw,
            # (P) Policy quality metrics
            "policy_entropy": avg_policy_entropy,
            "policy_top1_acc": policy_top1_acc,
            "policy_top3_acc": policy_top3_acc,
            # (C) Value confidence calibration
            "value_confidence_acc": value_confidence_acc,
            "value_confident_frac": value_confident_frac,
            # (RB) Per-residual-block gradient norms
            "rb_grad_norms": avg_rb_grad_norms,
            # Value target histogram
            "val_hist": val_hist,
            # 3-in-a-row target bias
            "three_r_diag": three_r_diag,
            # (#2) Per-iteration value loss delta
            "pre_train_vloss": pre_train_vloss,
            "vloss_delta": (val_vloss - pre_train_vloss) if pre_train_vloss is not None else None,
            # (#6) Value loss by game phase
            "phase_vloss_early": phase_vloss_sums['early'] / max(phase_counts['early'], 1),
            "phase_vloss_mid": phase_vloss_sums['mid'] / max(phase_counts['mid'], 1),
            "phase_vloss_late": phase_vloss_sums['late'] / max(phase_counts['late'], 1),
            "phase_counts": phase_counts,
            # (#8) Policy loss on value-critical positions
            "policy_loss_decisive": ploss_decisive_sum / max(decisive_count, 1),
            "policy_loss_ambiguous": ploss_ambiguous_sum / max(ambiguous_count, 1),
            "decisive_frac": decisive_count / max(decisive_count + ambiguous_count, 1),
            # Ownership auxiliary loss
            "avg_ownership_loss": avg_ownership_loss,
            # Symmetry loss (zero-sum constraint)
            "avg_symmetry_loss": avg_symmetry_loss,
            # Sub-iteration dynamics
            "sub_iter_log": sub_iter_log,
            # Value confidence distribution
            "conf_dist": {k: v / max(conf_total, 1) for k, v in conf_buckets.items()},
            # Intra-iteration value trajectory on FixedEval positions
            "fixed_eval_trajectory": fixed_eval_trajectory,
            # BUFFER_BIAS: per-player target distribution in buffer
            "buf_n_x": n_x_buf, "buf_n_o": n_o_buf,
            "buf_mean_tgt_x": mean_tgt_x, "buf_mean_tgt_o": mean_tgt_o,
            "buf_frac_pos_x": frac_pos_x, "buf_frac_pos_o": frac_pos_o,
            # PRED_BY_PLAYER: pre/post training predictions by player
            "pbias_pre_x_pred": _pbias_data['pre_x_pred'] if _pbias_data else 0.0,
            "pbias_pre_o_pred": _pbias_data['pre_o_pred'] if _pbias_data else 0.0,
            "pbias_pre_x_acc": _pbias_data['pre_x_acc'] if _pbias_data else 0.0,
            "pbias_pre_o_acc": _pbias_data['pre_o_acc'] if _pbias_data else 0.0,
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
        for iteration in range(num_iterations):
            iter_t0 = time.time()

            # Self-play
            t0 = time.time()
            all_examples, results, game_lengths = self._self_play(iteration)
            self_play_time = time.time() - t0

            # Augment with symmetries (e.g. left-right mirror for Connect4)
            augmented = []
            for ex in all_examples:
                ownership = ex[3] if len(ex) > 3 else None
                for sym_input, sym_policy, sym_own in self.game.get_symmetries(ex[0], ex[1], ownership):
                    entry = [sym_input, sym_policy, ex[2]]
                    if sym_own is not None:
                        entry.append(sym_own)
                    augmented.append(entry)
            self.buffer.insert_batch(augmented)

            # === Per-iteration 3-in-a-row target bias (fresh batch only) ===
            try:
                _iter_inputs = np.array([ex[0] for ex in augmented])  # (N, 2, H, W)
                _iter_targets = np.array([ex[2] for ex in augmented])  # (N,)
                _iter_3r = {}
                for _ch, _cname in [(0, 'mine'), (1, 'opp')]:
                    _cd = _iter_inputs[:, _ch]
                    _h3 = _cd[:, :, :-2] * _cd[:, :, 1:-1] * _cd[:, :, 2:]
                    _v3 = _cd[:, :-2, :] * _cd[:, 1:-1, :] * _cd[:, 2:, :]
                    _m = _h3.any(axis=(1, 2)) | _v3.any(axis=(1, 2))
                    _cnt = int(_m.sum())
                    _mt = float(_iter_targets[_m].mean()) if _cnt > 0 else 0.0
                    _iter_3r[_cname] = {'count': _cnt, 'mean': _mt}
                _mine = _iter_3r['mine']
                _opp = _iter_3r['opp']
                print(f"  Diag[3R-iter]: fresh batch ch0: n={_mine['count']} target={_mine['mean']:+.3f} | "
                      f"ch1: n={_opp['count']} target={_opp['mean']:+.3f}")
                self.writer.add_scalar("diag/iter_3r_ch0_target", _mine['mean'], iteration)
                self.writer.add_scalar("diag/iter_3r_ch1_target", _opp['mean'], iteration)
                # Sign split for this iteration's ch0 3-in-a-row
                _cd0 = _iter_inputs[:, 0]
                _h3_0 = _cd0[:, :, :-2] * _cd0[:, :, 1:-1] * _cd0[:, :, 2:]
                _v3_0 = _cd0[:, :-2, :] * _cd0[:, 1:-1, :] * _cd0[:, 2:, :]
                _ch0m = _h3_0.any(axis=(1, 2)) | _v3_0.any(axis=(1, 2))
                if _ch0m.any():
                    _t0 = _iter_targets[_ch0m]
                    print(f"  Diag[SIGN-iter]: ch0_3row: +:{int((_t0>0).sum())} -:{int((_t0<0).sum())} "
                          f"0:{int((_t0==0).sum())} (n={len(_t0)})")
            except Exception:
                pass

            # Log self-play stats
            wins_p1 = results.count(-1)
            wins_p2 = results.count(1)
            draws = results.count(0)
            avg_length = np.mean(game_lengths)
            min_length = int(np.min(game_lengths))
            max_length = int(np.max(game_lengths))
            p1_win_pct = wins_p1 / max(len(results), 1)
            self.writer.add_scalar("self_play/avg_game_length", avg_length, iteration)
            self.writer.add_scalar("self_play/wins_p1", wins_p1, iteration)
            self.writer.add_scalar("self_play/wins_p2", wins_p2, iteration)
            self.writer.add_scalar("self_play/draws", draws, iteration)
            self.writer.add_scalar("self_play/p1_win_pct", p1_win_pct, iteration)
            self.writer.add_scalar("self_play/buffer_size",
                                   sum(1 for s in self.buffer.arr if s is not None), iteration)

            # === Pre-training diagnostics ===
            self._eval_diagnostic_positions(iteration, prefix="pre_", label="PreTrainEval")

            # Pre-training segregation (weight-based, no forward pass needed)
            try:
                _val_w = self.net.value_conv.weight.data
                _pol_w = self.net.policy_conv.weight.data
                _val_ch = _val_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()
                _pol_ch = _pol_w.abs().sum(dim=(0, 2, 3)).cpu().numpy()
                pre_vp_corr = float(np.corrcoef(_val_ch, _pol_ch)[0, 1])
                _pre_top20_v = np.argsort(_val_ch)[-20:]
                _pre_top20_p = np.argsort(_pol_ch)[-20:]
                pre_overlap = len(set(_pre_top20_v.tolist()) & set(_pre_top20_p.tolist()))
                print(f"  PreSeg: vp_corr={pre_vp_corr:.3f} overlap={pre_overlap}/20")
                self.writer.add_scalar("pre_seg/vp_corr", pre_vp_corr, iteration)
                self.writer.add_scalar("pre_seg/overlap_20", pre_overlap, iteration)
            except Exception:
                pass

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
                    # Run through backbone only (conv + res_blocks + final_bn)
                    with torch.no_grad():
                        _x = F.relu(self.net.bn(ws_conv2d(_diag_inputs, self.net.conv)))
                        for block in self.net.res_blocks:
                            _x = block(_x)
                        _x = F.relu(self.net.final_bn(_x))
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
            except Exception:
                pass

            # Train
            t0 = time.time()
            train_result = self.train_network()
            train_time = time.time() - t0

            # === Post-training diagnostics: backbone drift + weight delta ===
            # (1) Backbone feature drift: compare backbone features on same positions
            try:
                if _pre_bb_features is not None and _diag_positions:
                    self.net.eval()
                    with torch.no_grad():
                        _x2 = F.relu(self.net.bn(ws_conv2d(_diag_inputs, self.net.conv)))
                        for block in self.net.res_blocks:
                            _x2 = block(_x2)
                        _x2 = F.relu(self.net.final_bn(_x2))
                        _post_bb_features = _x2.flatten(1).cpu()  # [5, channels*H*W]
                    # Per-position cosine similarity
                    _cos_sims = F.cosine_similarity(_pre_bb_features, _post_bb_features, dim=1)
                    _cos_mean = float(_cos_sims.mean())
                    _cos_min = float(_cos_sims.min())
                    _pos_names = [p[0] for p in _diag_positions]
                    _cos_detail = " ".join(
                        f"{_pos_names[i]}={float(_cos_sims[i]):.4f}"
                        for i in range(len(_cos_sims))
                    )
                    print(f"  Diag[DRIFT]: backbone cosine_sim mean={_cos_mean:.4f} "
                          f"min={_cos_min:.4f} | {_cos_detail}")
                    self.writer.add_scalar("drift/bb_cosine_mean", _cos_mean, iteration)
                    self.writer.add_scalar("drift/bb_cosine_min", _cos_min, iteration)
                    for i, nm in enumerate(_pos_names):
                        self.writer.add_scalar(f"drift/{nm}_cosine", float(_cos_sims[i]), iteration)
                    self.net.train()
            except Exception as e:
                print(f"  [DIAG-DBG] Backbone drift measurement failed: {e}")

            # (3) Per-block weight delta magnitude: ||w_after - w_before|| / ||w_before||
            try:
                if _pre_block_weights:
                    _wd_parts = []
                    for bi, block in enumerate(self.net.res_blocks):
                        _w_after = block.conv2.weight.data
                        _w_before = _pre_block_weights[bi]
                        _delta_norm = float((_w_after - _w_before).norm().item())
                        _before_norm = float(_w_before.norm().item())
                        _rel_delta = _delta_norm / max(_before_norm, 1e-10)
                        _wd_parts.append(f"rb{bi}={_rel_delta:.4f}")
                        self.writer.add_scalar(f"wdelta/rb{bi}_rel", _rel_delta, iteration)
                    print(f"  Diag[WDELTA]: {' '.join(_wd_parts)}")
            except Exception as e:
                print(f"  [DIAG-DBG] Weight delta measurement failed: {e}")

            iter_time = time.time() - iter_t0

            if train_result is not None:
                avg_loss, avg_value_loss, avg_policy_loss = train_result
                self.writer.add_scalar("loss/total", avg_loss, iteration)
                self.writer.add_scalar("loss/value", avg_value_loss, iteration)
                self.writer.add_scalar("loss/policy", avg_policy_loss, iteration)
                own_loss_str = ""
                if hasattr(self, '_train_diag') and self._train_diag.get("avg_ownership_loss", 0) > 0:
                    own_l = self._train_diag["avg_ownership_loss"]
                    self.writer.add_scalar("loss/ownership", own_l, iteration)
                    own_loss_str = f" o={own_l:.4f}"
                sym_loss_str = ""
                if hasattr(self, '_train_diag') and self._train_diag.get("avg_symmetry_loss", 0) > 0:
                    sym_l = self._train_diag["avg_symmetry_loss"]
                    self.writer.add_scalar("loss/symmetry", sym_l, iteration)
                    sym_loss_str = f" sym={sym_l:.4f}"
                print(f"  Iter {iteration+1}/{num_iterations}: loss={avg_loss:.4f} "
                      f"(v={avg_value_loss:.4f} p={avg_policy_loss:.4f}{own_loss_str}{sym_loss_str}) | "
                      f"games: p1={wins_p1} p2={wins_p2} draw={draws} | "
                      f"avg_len={avg_length:.1f} ({min_length}-{max_length}) | "
                      f"self_play={self_play_time:.1f}s train={train_time:.1f}s "
                      f"total={iter_time:.1f}s")

            self.writer.add_scalar("perf/self_play_time", self_play_time, iteration)
            self.writer.add_scalar("perf/train_time", train_time, iteration)
            if hasattr(self, '_batched') and hasattr(self._batched, 'perf'):
                perf = self._batched.perf
                avg_batch = perf["sample_count"] / max(perf["batch_count"], 1)
                self.writer.add_scalar("perf/mcts_select_expand", perf["select_expand_time"], iteration)
                self.writer.add_scalar("perf/mcts_backup", perf["backup_time"], iteration)
                self.writer.add_scalar("perf/nn_time", perf["nn_time"], iteration)
                self.writer.add_scalar("perf/nn_preprocess", perf["preprocess_time"], iteration)
                self.writer.add_scalar("perf/nn_transfer", perf["transfer_time"], iteration)
                self.writer.add_scalar("perf/nn_forward", perf["forward_time"], iteration)
                self.writer.add_scalar("perf/nn_result", perf["result_time"], iteration)
                self.writer.add_scalar("perf/nn_postprocess", perf["postprocess_time"], iteration)
                self.writer.add_scalar("perf/batch_count", perf["batch_count"], iteration)
                self.writer.add_scalar("perf/avg_batch_size", avg_batch, iteration)
                self.writer.add_scalar("perf/terminal_hits", perf["terminal_hits"], iteration)
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
                # Batch size histogram
                hist = perf.get("batch_histogram", [0]*5)
                print(f"  BatchHist: [1-4]={hist[0]} [5-16]={hist[1]} "
                      f"[17-32]={hist[2]} [33-64]={hist[3]} [65+]={hist[4]}")
                # Active games per move step
                apm = perf.get("active_per_move", [])
                if apm:
                    print(f"  ActiveGames: min={min(apm)} avg={np.mean(apm):.1f} "
                          f"max={max(apm)} steps={len(apm)}")
                # Accumulation rounds
                accum = perf.get("accum_rounds", 0)
                if accum > 0:
                    print(f"  Accumulation: {accum} move-steps used batch accumulation")
                # Tree reuse stats
                tr_count = perf.get("tree_reuse_count", 0)
                tr_fresh = perf.get("tree_reuse_fresh_count", 0)
                tr_total = tr_count + tr_fresh
                if tr_total > 0:
                    tr_pct = tr_count / tr_total
                    tr_avg_v = perf.get("tree_reuse_avg_visits", 0)
                    print(f"  TreeReuse: {tr_count}/{tr_total} ({tr_pct:.0%}) "
                          f"avg_reused_visits={tr_avg_v:.1f}")
                    self.writer.add_scalar("perf/tree_reuse_pct", tr_pct, iteration)
                    self.writer.add_scalar("perf/tree_reuse_avg_visits", tr_avg_v, iteration)
                # Resign stats
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
                    self.writer.add_scalar("self_play/resign_count", resign_count, iteration)
                    self.writer.add_scalar("self_play/resign_pct", resign_pct, iteration)
                    self.writer.add_scalar("self_play/resign_avg_move", resign_avg_move, iteration)
                    if resign_checks > 0:
                        fp_rate = resign_fp / resign_checks
                        self.writer.add_scalar("self_play/resign_fp_rate", fp_rate, iteration)
                # Immediate-win stats
                imm_win_n = perf.get("imm_win_count", 0)
                imm_win_frac = perf.get("imm_win_frac", 0)
                imm_win_total = imm_win_n / max(imm_win_frac, 1e-9) if imm_win_frac > 0 else 0
                print(f"  ImmWin: {imm_win_n}/{int(imm_win_total)} ({imm_win_frac:.1%}) positions have immediate winning move")
                self.writer.add_scalar("self_play/imm_win_frac", imm_win_frac, iteration)
                self.writer.add_scalar("self_play/imm_win_count", imm_win_n, iteration)
            if hasattr(self, '_train_perf'):
                tp = self._train_perf
                self.writer.add_scalar("perf/train_data_prep", tp["data_prep_time"], iteration)
                self.writer.add_scalar("perf/train_gradient", tp["gradient_time"], iteration)
                self.writer.add_scalar("perf/train_num_samples", tp["num_samples"], iteration)
                self.writer.add_scalar("perf/train_num_batches", tp["num_batches"], iteration)
                print(f"  Train: data={tp['data_prep_time']:.1f}s "
                      f"grad={tp['gradient_time']:.1f}s | "
                      f"samples={tp['num_samples']} "
                      f"batches={tp['num_batches']}")
            if hasattr(self, '_train_diag'):
                d = self._train_diag
                # Tensorboard
                self.writer.add_scalar("diag/val_target_mean", d["val_target_mean"], iteration)
                self.writer.add_scalar("diag/val_target_std", d["val_target_std"], iteration)
                self.writer.add_scalar("diag/frac_xwins", d["frac_neg"], iteration)
                self.writer.add_scalar("diag/frac_owins", d["frac_pos"], iteration)
                self.writer.add_scalar("diag/frac_draws", d["frac_draw"], iteration)
                self.writer.add_scalar("diag/effective_epochs", d["effective_epochs"], iteration)
                self.writer.add_scalar("diag/early_vloss", d["early_vloss"], iteration)
                self.writer.add_scalar("diag/late_vloss", d["late_vloss"], iteration)
                self.writer.add_scalar("diag/early_ploss", d["early_ploss"], iteration)
                self.writer.add_scalar("diag/late_ploss", d["late_ploss"], iteration)
                self.writer.add_scalar("diag/buffer_fill", d["buffer_fill"], iteration)
                self.writer.add_scalar("diag/pred_v_mean", d["pred_v_mean"], iteration)
                self.writer.add_scalar("diag/pred_v_std", d["pred_v_std"], iteration)
                self.writer.add_scalar("diag/pred_v_abs_mean", d["pred_v_abs_mean"], iteration)
                self.writer.add_scalar("diag/policy_grad_frac", d["policy_grad_frac"], iteration)
                self.writer.add_scalar("diag/val_loss_floor", d["val_loss_floor"], iteration)
                self.writer.add_scalar("diag/value_grad_norm", d["avg_value_grad_norm"], iteration)
                self.writer.add_scalar("diag/policy_grad_norm", d["avg_policy_grad_norm"], iteration)
                # New metrics: policy quality, value confidence, per-block grads
                self.writer.add_scalar("diag/policy_entropy", d["policy_entropy"], iteration)
                self.writer.add_scalar("diag/policy_top1_acc", d["policy_top1_acc"], iteration)
                self.writer.add_scalar("diag/policy_top3_acc", d["policy_top3_acc"], iteration)
                self.writer.add_scalar("diag/value_confidence_acc", d["value_confidence_acc"], iteration)
                self.writer.add_scalar("diag/value_confident_frac", d["value_confident_frac"], iteration)
                for rb_i, rb_gn in d.get("rb_grad_norms", {}).items():
                    self.writer.add_scalar(f"diag/rb{rb_i}_grad_norm", rb_gn, iteration)
                # (#2) Value loss delta
                if d.get("vloss_delta") is not None:
                    self.writer.add_scalar("diag/vloss_delta", d["vloss_delta"], iteration)
                    self.writer.add_scalar("diag/pre_train_vloss", d["pre_train_vloss"], iteration)
                # (#6) Game phase value loss
                self.writer.add_scalar("diag/phase_vloss_early", d["phase_vloss_early"], iteration)
                self.writer.add_scalar("diag/phase_vloss_mid", d["phase_vloss_mid"], iteration)
                self.writer.add_scalar("diag/phase_vloss_late", d["phase_vloss_late"], iteration)
                # (#8) Policy loss on decisive positions
                self.writer.add_scalar("diag/policy_loss_decisive", d["policy_loss_decisive"], iteration)
                self.writer.add_scalar("diag/policy_loss_ambiguous", d["policy_loss_ambiguous"], iteration)
                self.writer.add_scalar("diag/decisive_frac", d["decisive_frac"], iteration)
                # Console
                print(f"  Diag: targets mean={d['val_target_mean']:+.3f} "
                      f"std={d['val_target_std']:.3f} | "
                      f"X={d['frac_neg']:.1%} O={d['frac_pos']:.1%} "
                      f"draw={d['frac_draw']:.1%}")
                overfit_gap = d['val_vloss'] - d['late_vloss']
                self.writer.add_scalar("diag/overfit_gap_vloss", overfit_gap, iteration)
                vloss_delta_str = f" delta={d['vloss_delta']:+.4f}" if d.get('vloss_delta') is not None else ""
                print(f"  Diag: eff_epochs={d['effective_epochs']:.1f} "
                      f"vlw={d.get('effective_vlw',1.0):.2f} "
                      f"steps={d['num_steps']} | "
                      f"vloss train={d['late_vloss']:.4f} "
                      f"val={d['val_vloss']:.4f} "
                      f"(gap={overfit_gap:+.4f}){vloss_delta_str} | "
                      f"buf={d['buffer_fill']}/{d['buffer_capacity']}"
                      f"{' FULL' if d['buffer_full'] else ''}")
                self.writer.add_scalar("diag/val_vloss", d["val_vloss"], iteration)
                self.writer.add_scalar("diag/effective_vlw", d.get("effective_vlw", 1.0), iteration)
                self.writer.add_scalar("diag/val_ploss", d["val_ploss"], iteration)
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
                # (P) Policy quality metrics
                print(f"  Diag[P]: entropy={d['policy_entropy']:.3f} "
                      f"top1_acc={d['policy_top1_acc']:.1%} "
                      f"top3_acc={d['policy_top3_acc']:.1%}")
                # (#8) Policy loss on decisive vs ambiguous
                print(f"  Diag[P2]: ploss_decisive={d['policy_loss_decisive']:.4f} "
                      f"ploss_ambiguous={d['policy_loss_ambiguous']:.4f} "
                      f"decisive_frac={d['decisive_frac']:.1%}")
                # (C) Value confidence calibration
                print(f"  Diag[C]: confident_acc={d['value_confidence_acc']:.1%} "
                      f"(frac_confident={d['value_confident_frac']:.1%})")
                # (RB) Per-block gradient norms
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
                # Value target histogram
                vh_bins = d.get('val_hist', [])
                if vh_bins:
                    print(f"  Diag[TH]: targets [-1,-0.5)={vh_bins[0]:.1%} "
                          f"[-0.5,0)={vh_bins[1]:.1%} [0]={vh_bins[2]:.1%} "
                          f"(0,0.5]={vh_bins[3]:.1%} (0.5,1]={vh_bins[4]:.1%}")
                # 3-in-a-row target bias
                tr = d.get('three_r_diag', {})
                if tr:
                    mine = tr.get('mine', {})
                    opp = tr.get('opp', {})
                    print(f"  Diag[3R]: ch0_3row: n={mine.get('count',0)} target={mine.get('mean_target',0):+.3f} "
                          f"({mine.get('frac',0):.1%}) | "
                          f"ch1_3row: n={opp.get('count',0)} target={opp.get('mean_target',0):+.3f} "
                          f"({opp.get('frac',0):.1%})")
                    self.writer.add_scalar("diag/three_r_ch0_target", mine.get('mean_target', 0), iteration)
                    self.writer.add_scalar("diag/three_r_ch1_target", opp.get('mean_target', 0), iteration)
                    self.writer.add_scalar("diag/three_r_ch0_frac", mine.get('frac', 0), iteration)
                    self.writer.add_scalar("diag/three_r_ch1_frac", opp.get('frac', 0), iteration)
                # (A) Per-player value breakdown
                print(f"  Diag[A]: X_vloss={d['x_vloss']:.4f} O_vloss={d['o_vloss']:.4f} | "
                      f"X_target={d['x_target_mean']:+.3f} O_target={d['o_target_mean']:+.3f} | "
                      f"X_pred={d['x_pred_mean']:+.3f} O_pred={d['o_pred_mean']:+.3f}")
                # BUFFER_BIAS: per-player target distribution in buffer
                print(f"  Diag[BufBias]: X-to-move: n={d.get('buf_n_x',0)} "
                      f"mean_tgt={d.get('buf_mean_tgt_x',0):+.3f} "
                      f"frac_pos={d.get('buf_frac_pos_x',0):.1%} | "
                      f"O-to-move: n={d.get('buf_n_o',0)} "
                      f"mean_tgt={d.get('buf_mean_tgt_o',0):+.3f} "
                      f"frac_pos={d.get('buf_frac_pos_o',0):.1%}")
                self.writer.add_scalar("pbias/buf_mean_tgt_x", d.get('buf_mean_tgt_x', 0), iteration)
                self.writer.add_scalar("pbias/buf_mean_tgt_o", d.get('buf_mean_tgt_o', 0), iteration)
                self.writer.add_scalar("pbias/buf_frac_pos_x", d.get('buf_frac_pos_x', 0), iteration)
                self.writer.add_scalar("pbias/buf_frac_pos_o", d.get('buf_frac_pos_o', 0), iteration)
                # PRED_BY_PLAYER: pre/post training predictions split by current player
                print(f"  Diag[PBias]: pre: X_pred={d.get('pbias_pre_x_pred',0):+.3f} "
                      f"X_acc={d.get('pbias_pre_x_acc',0):.1%} | "
                      f"O_pred={d.get('pbias_pre_o_pred',0):+.3f} "
                      f"O_acc={d.get('pbias_pre_o_acc',0):.1%}")
                print(f"  Diag[PBias]: post: X_pred={d.get('pbias_post_x_pred',0):+.3f} "
                      f"X_acc={d.get('pbias_post_x_acc',0):.1%} | "
                      f"O_pred={d.get('pbias_post_o_pred',0):+.3f} "
                      f"O_acc={d.get('pbias_post_o_acc',0):.1%}")
                self.writer.add_scalar("pbias/pre_x_pred", d.get('pbias_pre_x_pred', 0), iteration)
                self.writer.add_scalar("pbias/pre_o_pred", d.get('pbias_pre_o_pred', 0), iteration)
                self.writer.add_scalar("pbias/post_x_pred", d.get('pbias_post_x_pred', 0), iteration)
                self.writer.add_scalar("pbias/post_o_pred", d.get('pbias_post_o_pred', 0), iteration)
                self.writer.add_scalar("pbias/pre_x_acc", d.get('pbias_pre_x_acc', 0), iteration)
                self.writer.add_scalar("pbias/pre_o_acc", d.get('pbias_pre_o_acc', 0), iteration)
                self.writer.add_scalar("pbias/post_x_acc", d.get('pbias_post_x_acc', 0), iteration)
                self.writer.add_scalar("pbias/post_o_acc", d.get('pbias_post_o_acc', 0), iteration)
                # (#6) Value loss by game phase
                pc = d.get('phase_counts', {})
                print(f"  Diag[GP]: vloss early={d['phase_vloss_early']:.4f}({pc.get('early',0)}) "
                      f"mid={d['phase_vloss_mid']:.4f}({pc.get('mid',0)}) "
                      f"late={d['phase_vloss_late']:.4f}({pc.get('late',0)})")
                # (F) Gradient direction
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
                # (V) Value head health
                vh = d.get('vh_diag', {})
                if vh:
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
                        self.writer.add_scalar("diag/bb_grad_cosine_sim", vh['bb_grad_cosine_sim'], iteration)
                        self.writer.add_scalar("diag/bb_grad_conflict_channels", vh['bb_grad_conflict_channels'], iteration)
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
                        self.writer.add_scalar("diag/final_bn_eff_gain_mean", vh['final_bn_eff_gain_mean'], iteration)
                        self.writer.add_scalar("diag/final_bn_eff_gain_min", vh['final_bn_eff_gain_min'], iteration)
                        self.writer.add_scalar("diag/final_bn_dead", vh['final_bn_dead'], iteration)
                        self.writer.add_scalar("diag/final_bn_gamma_mean", vh.get('final_bn_gamma_mean', 0), iteration)
                        self.writer.add_scalar("diag/final_bn_gamma_std", vh.get('final_bn_gamma_std', 0), iteration)
                        self.writer.add_scalar("diag/final_bn_sqrt_var_mean", vh.get('final_bn_sqrt_var_mean', 0), iteration)
                        self.writer.add_scalar("diag/final_bn_sqrt_var_std", vh.get('final_bn_sqrt_var_std', 0), iteration)
                    # (#5) Residual contribution ratio
                    rr = vh.get('rb_residual_ratios', {})
                    if rr:
                        rr_str = " ".join(f"rb{k}={v:.3f}" for k, v in sorted(rr.items()))
                        print(f"  Diag[RR]: residual_ratio: {rr_str}")
                        for bi, ratio in rr.items():
                            self.writer.add_scalar(f"rb{bi}/residual_ratio", ratio, iteration)
                    # (#3) Initial conv weight norm
                    if 'init_conv_w_norm' in vh:
                        print(f"  Diag[IC]: conv_w norm={vh['init_conv_w_norm']:.4f} "
                              f"|w|={vh['init_conv_w_abs_mean']:.5f} "
                              f"delta={vh['init_conv_w_norm_delta']:+.4f}")
                        self.writer.add_scalar("diag/init_conv_w_norm", vh['init_conv_w_norm'], iteration)
                    # Tensorboard
                    self.writer.add_scalar("vh/dead_neurons", vh["dead_neurons"], iteration)
                    self.writer.add_scalar("vh/active_neurons", vh.get("active_neurons", 0), iteration)
                    # WDL metrics (replaces pre_tanh/saturated)
                    self.writer.add_scalar("vh/wdl_entropy", vh["wdl_entropy"], iteration)
                    self.writer.add_scalar("vh/wdl_confidence", vh["wdl_confidence"], iteration)
                    self.writer.add_scalar("vh/wdl_accuracy", vh["wdl_accuracy"], iteration)
                    self.writer.add_scalar("vh/wdl_logit_std", vh["wdl_logit_std"], iteration)
                    self.writer.add_scalar("vh/wdl_logit_range", vh["wdl_logit_range"], iteration)
                    self.writer.add_scalar("vh/wdl_win_prob", vh["wdl_win_prob"], iteration)
                    self.writer.add_scalar("vh/wdl_draw_prob", vh["wdl_draw_prob"], iteration)
                    self.writer.add_scalar("vh/wdl_loss_prob", vh["wdl_loss_prob"], iteration)
                    self.writer.add_scalar("vh/wdl_scalar_mean", vh["wdl_scalar_mean"], iteration)
                    self.writer.add_scalar("vh/wdl_scalar_std", vh["wdl_scalar_std"], iteration)
                    self.writer.add_scalar("vh/fc2_w_max", vh["fc2_w_max"], iteration)
                    self.writer.add_scalar("vh/fc2_w_min", vh["fc2_w_min"], iteration)
                    self.writer.add_scalar("vh/fc2_w_norm", vh["fc2_w_norm"], iteration)
                    self.writer.add_scalar("vh/fc1_w_norm", vh["fc1_w_norm"], iteration)
                    self.writer.add_scalar("vh/backbone_std", vh["backbone_std"], iteration)
                    self.writer.add_scalar("vh/fc1_act_p10", vh["fc1_act_p10"], iteration)
                    self.writer.add_scalar("vh/fc1_act_p50", vh["fc1_act_p50"], iteration)
                    self.writer.add_scalar("vh/fc1_act_p90", vh["fc1_act_p90"], iteration)
                    self.writer.add_scalar("vh/vconv_dead_channels", vh["vconv_dead_channels"], iteration)
                    self.writer.add_scalar("vh/fc2_w_norm_delta", vh["fc2_w_norm_delta"], iteration)
                    self.writer.add_scalar("vh/fc1_w_norm_delta", vh["fc1_w_norm_delta"], iteration)
                    self.writer.add_scalar("vh/grad_dead_mean", vh["grad_dead_mean"], iteration)
                    self.writer.add_scalar("vh/grad_alive_mean", vh["grad_alive_mean"], iteration)
                    self.writer.add_scalar("vc/backbone_raw_abs", vh["backbone_raw_abs"], iteration)
                    self.writer.add_scalar("vc/vconv_pre_bn_abs", vh["vconv_pre_bn_abs"], iteration)
                    self.writer.add_scalar("vc/vbn_post_abs", vh["vbn_post_abs"], iteration)
                    self.writer.add_scalar("vc/bn_ratio", vh["bn_ratio"], iteration)
                    self.writer.add_scalar("vc/vc_w_norm", vh["vc_w_norm"], iteration)
                    self.writer.add_scalar("vc/vbn_gamma_min", vh["vbn_gamma_min"], iteration)
                    self.writer.add_scalar("vc/vbn_gamma_mean", vh["vbn_gamma_mean"], iteration)
                    self.writer.add_scalar("vc/pconv_abs", vh["pconv_abs"], iteration)
                    self.writer.add_scalar("vc/vconv_grad_norm", vh["vconv_grad_norm"], iteration)
                    # Backbone per-channel stats
                    if 'bb_n_channels' in vh:
                        self.writer.add_scalar("bb/dead_channels", vh["bb_dead_channels"], iteration)
                        self.writer.add_scalar("bb/ch_p10", vh["bb_ch_p10"], iteration)
                        self.writer.add_scalar("bb/ch_p50", vh["bb_ch_p50"], iteration)
                        self.writer.add_scalar("bb/ch_p90", vh["bb_ch_p90"], iteration)
                    # Backbone gradient decomposition
                    if 'bb_v_grad_norm' in vh:
                        self.writer.add_scalar("bb/v_grad_norm", vh["bb_v_grad_norm"], iteration)
                        self.writer.add_scalar("bb/p_grad_norm", vh["bb_p_grad_norm"], iteration)
                        self.writer.add_scalar("bb/grad_ratio", vh["bb_grad_ratio"], iteration)
                        self.writer.add_scalar("bb/n_value_dom", vh["bb_n_value_dom"], iteration)
                        self.writer.add_scalar("bb/n_policy_dom", vh["bb_n_policy_dom"], iteration)
                    if 'bb_param_v_grad' in vh:
                        self.writer.add_scalar("bb/param_v_grad", vh["bb_param_v_grad"], iteration)
                        self.writer.add_scalar("bb/param_p_grad", vh["bb_param_p_grad"], iteration)
                        self.writer.add_scalar("bb/param_grad_ratio", vh["bb_param_grad_ratio"], iteration)
                    if 'vp_weight_corr' in vh:
                        self.writer.add_scalar("vp/weight_corr", vh["vp_weight_corr"], iteration)
                        self.writer.add_scalar("vp/overlap_20", vh["vp_overlap_20"], iteration)
                        self.writer.add_scalar("vp/val_top20_act", vh["vp_val_top20_act"], iteration)
                        self.writer.add_scalar("vp/pol_top20_act", vh["vp_pol_top20_act"], iteration)
                        self.writer.add_scalar("vp/val_health_corr", vh["vp_val_health_corr"], iteration)
                        self.writer.add_scalar("vp/pol_health_corr", vh["vp_pol_health_corr"], iteration)
                    if 'svd_bb_rank90' in vh:
                        self.writer.add_scalar("svd/bb_rank90", vh["svd_bb_rank90"], iteration)
                        self.writer.add_scalar("svd/bb_rank99", vh["svd_bb_rank99"], iteration)
                        self.writer.add_scalar("svd/bb_near_zero", vh["svd_bb_near_zero"], iteration)
                        self.writer.add_scalar("svd/bn_dead_deepest", vh["bn_dead_deepest"], iteration)
                        self.writer.add_scalar("svd/pfc_rank90", vh["svd_pfc_rank90"], iteration)
                        self.writer.add_scalar("svd/pfc_rank99", vh["svd_pfc_rank99"], iteration)
                        for i, n in enumerate(vh['pconv_ch_norms']):
                            self.writer.add_scalar(f"ph/pconv_ch{i}_norm", n, iteration)
                    # Per-block BN and activation TensorBoard
                    for bi, rbd in vh.get('all_rb_bn', {}).items():
                        self.writer.add_scalar(f"rb{bi}/bn2_dead", rbd.get("dead", 0), iteration)
                        self.writer.add_scalar(f"rb{bi}/bn2_neg_gamma", rbd.get("neg_gamma", 0), iteration)
                        self.writer.add_scalar(f"rb{bi}/bn2_eff_gain_mean", rbd.get("eff_gain_mean", 0), iteration)
                        self.writer.add_scalar(f"rb{bi}/svd_rank90", rbd.get("svd_rank90", 0), iteration)
                        self.writer.add_scalar(f"rb{bi}/bn2_gamma_mean", rbd.get("gamma_mean", 0), iteration)
                        self.writer.add_scalar(f"rb{bi}/bn2_sqrt_var_mean", rbd.get("sqrt_var_mean", 0), iteration)
                    for bi, rad in vh.get('rb_act_stats', {}).items():
                        self.writer.add_scalar(f"rb{bi}/act_abs_mean", rad["abs_mean"], iteration)
                        self.writer.add_scalar(f"rb{bi}/act_std", rad["std"], iteration)
                        self.writer.add_scalar(f"rb{bi}/act_dead_channels", rad["dead_channels"], iteration)
                    # (#6) Per-conv weight norms
                    for bi, rcn in vh.get('rb_conv_norms', {}).items():
                        self.writer.add_scalar(f"rb{bi}/conv1_w_norm", rcn["conv1"], iteration)
                        self.writer.add_scalar(f"rb{bi}/conv2_w_norm", rcn["conv2"], iteration)
                        self.writer.add_scalar(f"rb{bi}/conv2_w_delta", rcn.get("c2_delta", 0), iteration)
                    # (#7-8) BN2 output scale + pre-BN2 variance
                    for bi, rbs in vh.get('rb_bn2_stats', {}).items():
                        self.writer.add_scalar(f"rb{bi}/bn2_out_abs", rbs["bn2_out_abs"], iteration)
                        self.writer.add_scalar(f"rb{bi}/bn2_out_std", rbs["bn2_out_std"], iteration)
                        if 'conv2_raw_var' in rbs:
                            self.writer.add_scalar(f"rb{bi}/conv2_raw_var", rbs["conv2_raw_var"], iteration)
                            self.writer.add_scalar(f"rb{bi}/conv2_raw_abs", rbs["conv2_raw_abs"], iteration)
                        if 'bn2_batch_vs_run_var_mean' in rbs:
                            self.writer.add_scalar(f"rb{bi}/bn2_bv_rv_ratio", rbs["bn2_batch_vs_run_var_mean"], iteration)
                            self.writer.add_scalar(f"rb{bi}/bn2_batch_var", rbs["bn2_batch_var_mean"], iteration)
                            self.writer.add_scalar(f"rb{bi}/bn2_run_var", rbs["bn2_run_var_mean"], iteration)
                    # (#11) Residual path effective rank
                    for bi, rrk in vh.get('rb_res_rank', {}).items():
                        self.writer.add_scalar(f"rb{bi}/res_rank90", rrk["rank90"], iteration)
                        self.writer.add_scalar(f"rb{bi}/res_rank99", rrk["rank99"], iteration)
                    # (#9) Per-block channel dominance
                    for bi, nv in vh.get('rb_ch_dominance', {}).items():
                        self.writer.add_scalar(f"rb{bi}/ch_value_dom", nv, iteration)
                    # (#10) Value gradient survival
                    for bn, sv in vh.get('rb_v_grad_survival', {}).items():
                        self.writer.add_scalar(f"rb{bn}/v_grad_survival", sv, iteration)

            # === Self-play value prediction diagnostics ===
            if hasattr(self, '_batched') and hasattr(self._batched, 'value_diag'):
                vd = self._batched.value_diag
                if vd:
                    self.writer.add_scalar("selfplay_diag/mean_nnet_value", vd["mean_nnet_value"], iteration)
                    self.writer.add_scalar("selfplay_diag/std_nnet_value", vd["std_nnet_value"], iteration)
                    self.writer.add_scalar("selfplay_diag/frac_saturated", vd["frac_saturated_any"], iteration)
                    self.writer.add_scalar("selfplay_diag/sign_accuracy", vd["sign_accuracy"], iteration)
                    self.writer.add_scalar("selfplay_diag/mae_vs_outcome", vd["mae_vs_outcome"], iteration)
                    self.writer.add_scalar("selfplay_diag/pred_outcome_corr", vd["pred_outcome_corr"], iteration)
                    self.writer.add_scalar("selfplay_diag/mean_when_x_moves", vd["mean_when_x_moves"], iteration)
                    self.writer.add_scalar("selfplay_diag/mean_when_o_moves", vd["mean_when_o_moves"], iteration)
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
                        self.writer.add_scalar("selfplay_diag/mcts_visit_entropy",
                                               vd["mcts_visit_entropy_mean"], iteration)
                        print(f"  SelfPlay: mcts_visit_entropy="
                              f"{vd['mcts_visit_entropy_mean']:.3f} "
                              f"(std={vd['mcts_visit_entropy_std']:.3f})")
                    # (#7) MCTS Q vs nnet value agreement
                    if 'mcts_nnet_corr' in vd:
                        self.writer.add_scalar("selfplay_diag/mcts_nnet_corr", vd["mcts_nnet_corr"], iteration)
                        self.writer.add_scalar("selfplay_diag/mcts_nnet_mae", vd["mcts_nnet_mae"], iteration)
                        self.writer.add_scalar("selfplay_diag/mcts_correction_mean", vd["mcts_correction_mean"], iteration)
                        print(f"  SelfPlay: mcts_Q mean={vd['mcts_q_mean']:+.3f} "
                              f"std={vd['mcts_q_std']:.3f} | "
                              f"nnet_Q_corr={vd['mcts_nnet_corr']:+.3f} "
                              f"MAE={vd['mcts_nnet_mae']:.3f} "
                              f"correction={vd['mcts_correction_mean']:+.3f}")

            # === Value confidence distribution ===
            if hasattr(self, '_train_diag'):
                cd = self._train_diag.get('conf_dist', {})
                if cd:
                    print(f"  Diag[CONF]: |v|<0.1={cd.get('very_low',0):.1%} "
                          f"0.1-0.3={cd.get('low',0):.1%} "
                          f"0.3-0.6={cd.get('medium',0):.1%} "
                          f"0.6-0.9={cd.get('high',0):.1%} "
                          f"|v|>0.9={cd.get('very_high',0):.1%}")
                    for bk, bv in cd.items():
                        self.writer.add_scalar(f"conf/{bk}", bv, iteration)

                # === Sub-iteration dynamics summary ===
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
                    self.writer.add_scalar("sub/vloss_first", sil[0]['vloss'], iteration)
                    self.writer.add_scalar("sub/vloss_last", sil[-1]['vloss'], iteration)
                    self.writer.add_scalar("sub/conf_first", sil[0]['mean_conf'], iteration)
                    self.writer.add_scalar("sub/conf_last", sil[-1]['mean_conf'], iteration)
                    self.writer.add_scalar("sub/mean_v_first", sil[0]['mean_v'], iteration)
                    self.writer.add_scalar("sub/mean_v_last", sil[-1]['mean_v'], iteration)

            # === Intra-iteration value trajectory on FixedEval positions ===
            if hasattr(self, '_train_diag'):
                _traj = self._train_diag.get('fixed_eval_trajectory', [])
                if _traj and len(_traj) >= 2:
                    # Print trajectory for each position: step0→step300→...→stepN
                    _names = [k for k in _traj[0] if k != 'step']
                    for _pn in _names:
                        _vals = [f"s{e['step']}:{e[_pn]:+.3f}" for e in _traj]
                        print(f"  Diag[TRAJ]: {_pn}: {' -> '.join(_vals)}")
                        # TB: log first and last values for trend detection
                        self.writer.add_scalar(f"traj/{_pn}_first", _traj[0][_pn], iteration)
                        self.writer.add_scalar(f"traj/{_pn}_last", _traj[-1][_pn], iteration)

            # === Fixed diagnostic position evaluation ===
            self._eval_diagnostic_positions(iteration)

            # Save every 5 iterations + always on the last one
            # Also save iteration 0 if no checkpoint exists (quick sanity check)
            no_checkpoint = not os.path.exists(os.path.join(self.checkpoint_dir, "latest.txt"))
            if (iteration + 1) % 5 == 0 or iteration == num_iterations - 1 or (iteration == 0 and no_checkpoint):
                self.net.save(self.checkpoint_dir, iteration=iteration, num_iterations=num_iterations)

        self.writer.close()

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
