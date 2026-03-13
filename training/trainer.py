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
        self.lr = self.config.get("lr", 0.001)
        self.device = self.config.get("device", "cpu")
        self.max_train_steps = self.config.get("max_train_steps", 5000)
        self.target_epochs = self.config.get("target_epochs", 4)
        self.buffer = ReplayBuffer(self.config.get("buffer_size", 100000))
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        self.value_loss_weight = self.config.get("value_loss_weight", 1.0)

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

        # Dynamic training steps: target N effective epochs, capped by max_train_steps
        # Scale epochs with buffer fill to prevent memorization when buffer is small
        n_samples = len(samples)
        buffer_capacity = self.buffer.max_size
        fill_ratio = min(n_samples / max(buffer_capacity, 1), 1.0)
        # Linearly scale from 1 epoch (empty) to target_epochs (full)
        scaled_epochs = 1.0 + (self.target_epochs - 1.0) * fill_ratio
        target_steps = int(scaled_epochs * (n_samples // self.batch_size))
        # Ramp value_loss_weight: 1.0 (empty) -> configured value (full)
        effective_vlw = 1.0 + (self.value_loss_weight - 1.0) * fill_ratio
        num_steps = max(1, min(self.max_train_steps, target_steps))
        effective_epochs = (num_steps * self.batch_size) / n_samples
        early_cutoff = max(num_steps // 10, 1)
        late_start = num_steps - early_cutoff

        # Cosine LR schedule: lr decays from initial to 10% over training steps
        lr_min = self.lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr  # reset to initial at start of each iteration

        for step in range(num_steps):
            # Cosine annealing: lr goes from self.lr -> lr_min
            lr = lr_min + 0.5 * (self.lr - lr_min) * (1 + math.cos(math.pi * step / num_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            batch = random.choices(samples, k=self.batch_size)

            t0 = time.time()
            states = torch.FloatTensor(np.array([s[0] for s in batch])).to(self.device)
            target_pis = torch.FloatTensor(np.array([s[1] for s in batch])).to(self.device)
            target_vs = torch.FloatTensor(np.array([s[2] for s in batch])).unsqueeze(1).to(self.device)
            data_prep_time += time.time() - t0

            t0 = time.time()
            pred_vs, pred_pis = self.net(states)

            value_loss = F.mse_loss(pred_vs, target_vs)
            policy_loss = -torch.mean(torch.sum(target_pis * torch.log(pred_pis + 1e-8), dim=1))
            loss = effective_vlw * value_loss + policy_loss

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

                    per_sample_vloss = (pred_vs - target_vs).pow(2).squeeze(1)
                    if is_x.any():
                        x_vloss_sum += per_sample_vloss[is_x].mean().item()
                        x_target_sum += target_vs[is_x].mean().item()
                        x_pred_sum += pred_vs[is_x].mean().item()
                        x_count += 1
                    if is_o.any():
                        o_vloss_sum += per_sample_vloss[is_o].mean().item()
                        o_target_sum += target_vs[is_o].mean().item()
                        o_pred_sum += pred_vs[is_o].mean().item()
                        o_count += 1

            # (F) Gradient direction check every 50 steps
            if step % 50 == 0:
                with torch.no_grad():
                    # Check: does the gradient want to push pred_v toward target_v?
                    # For MSE loss, grad w.r.t. pred = 2*(pred - target)
                    # Correct direction means: sign(pred - target) should guide update
                    # After optimizer.step(), pred should move toward target
                    # We check: does pred_v have same sign error pattern as grad?
                    error = (pred_vs - target_vs).squeeze(1)
                    # Gradient is 2*error, optimizer subtracts lr*grad
                    # So pred should decrease where error > 0, increase where error < 0
                    # This is always correct for MSE — but we want to verify the
                    # value head fc layers specifically get the right gradient sign
                    grad_correct_count += 1  # MSE always has correct direction
                    grad_total_count += 1
                    # More useful: check if value head weights actually update correctly
                    # Log the actual value head gradient statistics
                    fc1_grad = self.net.value_fc1.weight.grad
                    fc2_grad = self.net.value_fc2.weight.grad
                    if fc1_grad is not None and fc2_grad is not None:
                        # Store for later analysis
                        if not hasattr(self, '_grad_stats'):
                            self._grad_stats = []
                        # Per-neuron gradient norms for fc1 (each row = one neuron)
                        fc1_per_neuron_gnorm = fc1_grad.norm(dim=1).cpu().numpy()  # [neurons]
                        # Value conv + BN gradient norms
                        vconv_g = self.net.value_conv.weight.grad
                        vbn_g = self.net.value_bn.weight.grad  # gamma grad
                        self._grad_stats.append({
                            'fc1_grad_mean': fc1_grad.mean().item(),
                            'fc1_grad_std': fc1_grad.std().item(),
                            'fc1_grad_norm': fc1_grad.norm().item(),
                            'fc2_grad_mean': fc2_grad.mean().item(),
                            'fc2_grad_std': fc2_grad.std().item(),
                            'fc2_grad_norm': fc2_grad.norm().item(),
                            'pred_mean': pred_vs.mean().item(),
                            'target_mean': target_vs.mean().item(),
                            'error_mean': error.mean().item(),
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

            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            gradient_time += time.time() - t0

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
                all_pred_vs.append(pred_vs.detach().cpu().numpy().flatten())

                # (P) Policy quality metrics
                with torch.no_grad():
                    # Policy entropy: H = -sum(p * log(p))
                    log_pi = torch.log(pred_pis + 1e-8)
                    batch_entropy = -(pred_pis * log_pi).sum(dim=1).mean().item()
                    all_policy_entropy.append(batch_entropy)

                    # Policy top-1 accuracy: does argmax match?
                    pred_top = pred_pis.argmax(dim=1)
                    target_top = target_pis.argmax(dim=1)
                    top1_correct_sum += (pred_top == target_top).float().sum().item()

                    # Policy top-3 accuracy: is MCTS best move in network's top 3?
                    pred_top3 = pred_pis.topk(3, dim=1).indices
                    target_argmax = target_pis.argmax(dim=1).unsqueeze(1)
                    top3_correct_sum += (pred_top3 == target_argmax).any(dim=1).float().sum().item()

                    policy_acc_count += pred_pis.shape[0]

                    # (C) Value confidence calibration
                    confident_mask = pred_vs.abs() > 0.5
                    if confident_mask.any():
                        confident_signs_correct = (
                            pred_vs[confident_mask].sign() == target_vs[confident_mask].sign()
                        ).float()
                        confident_correct_sum += confident_signs_correct.sum().item()
                        confident_total += confident_mask.sum().item()

        avg_loss = total_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
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
                vt_v = torch.FloatTensor(np.array([s[2] for s in vb])).unsqueeze(1).to(self.device)
                pv, pp = self.net(vs)
                val_vloss += F.mse_loss(pv, vt_v).item()
                val_ploss += -torch.mean(torch.sum(vt_pi * torch.log(pp + 1e-8), dim=1)).item()
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

        # Theoretical value loss floor: E[(v - target)^2] when v=mean(targets)
        # For binary ±1 targets, floor = 1 - mean^2
        val_loss_floor = val_std ** 2  # variance of targets = irreducible MSE

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

        # (V) Value head health: dead neurons, pre-tanh range, saturation,
        # weight stats, spike decomposition, backbone signal
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
                captured['pre_tanh'] = output
            # Per-residual-block activation hooks
            def make_rb_hook(idx):
                def hook(module, input, output):
                    captured[f'rb{idx}_out'] = output
                return hook
            rb_hooks = []
            for idx, block in enumerate(self.net.res_blocks):
                rb_hooks.append(block.register_forward_hook(make_rb_hook(idx)))

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
                v_out, _ = self.net(diag_inp)
            h0.remove()
            h0b.remove()
            h0p.remove()
            h1.remove()
            h2.remove()
            for h in rb_hooks:
                h.remove()

            backbone_raw_np = captured['backbone_raw'].cpu().numpy()  # [batch, filters, H, W]
            vconv_np = captured['vconv_out'].cpu().numpy()   # [batch, ch, H, W] (pre-BN)
            vbn_np = captured['vbn_out'].cpu().numpy()       # [batch, ch, H, W] (post-BN)
            pconv_np = captured['pconv_out'].cpu().numpy()   # [batch, 2, H, W]
            fc1_np = captured['fc1_out'].cpu().numpy()       # [batch, neurons]
            fc2_in_np = captured['fc2_in'].cpu().numpy()     # [batch, neurons] post-activation
            pre_tanh_np = captured['pre_tanh'].cpu().numpy().flatten()
            out_np = v_out.cpu().numpy().flatten()
            fc1_in_np = captured['fc1_in'].cpu().numpy()     # [batch, flat_size]

            n_total_neurons = fc1_np.shape[1]
            # With LeakyReLU, "near-dead" = always near zero (abs < 0.01)
            near_dead = (np.abs(fc1_np) < 0.01).all(axis=0)
            n_dead = int(near_dead.sum())

            # --- fc2 weight stats: are weights growing? ---
            fc2_w = self.net.value_fc2.weight.detach().cpu().numpy().flatten()
            fc2_b = self.net.value_fc2.bias.detach().cpu().numpy().flatten()

            # --- fc1 weight stats ---
            fc1_w = self.net.value_fc1.weight.detach().cpu().numpy()

            # --- Spike decomposition: what causes max pre_tanh? ---
            max_idx = int(np.argmax(np.abs(pre_tanh_np)))
            max_sample_acts = fc2_in_np[max_idx]  # activations for spike sample
            contributions = max_sample_acts * fc2_w  # per-neuron contribution
            top_neuron = int(np.argmax(np.abs(contributions)))

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
                "pre_tanh_min": float(pre_tanh_np.min()),
                "pre_tanh_max": float(pre_tanh_np.max()),
                "pre_tanh_std": float(pre_tanh_np.std()),
                "out_saturated": float((np.abs(out_np) > 0.95).mean()),
                "out_abs_mean": float(np.abs(out_np).mean()),
                # fc2 weight diagnostics
                "fc2_w_max": float(fc2_w.max()),
                "fc2_w_min": float(fc2_w.min()),
                "fc2_w_norm": fc2_w_norm_now,
                "fc2_bias": float(fc2_b[0]),
                # fc1 weight diagnostics
                "fc1_w_norm": fc1_w_norm_now,
                # Spike decomposition
                "spike_neuron": top_neuron,
                "spike_contribution": float(contributions[top_neuron]),
                "spike_fc2_weight": float(fc2_w[top_neuron]),
                "spike_fc1_act": float(max_sample_acts[top_neuron]),
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
            }
            self.net.train()
        except Exception:
            pass

        # === Backbone gradient decomposition: value vs policy ===
        # Separate backward passes to see which head drives each backbone channel
        try:
            self.net.eval()  # Don't update BN running stats
            gd_batch = random.choices(samples, k=min(64, len(samples)))
            gd_states = torch.FloatTensor(np.array([s[0] for s in gd_batch])).to(self.device)
            gd_targets_v = torch.FloatTensor(np.array([s[2] for s in gd_batch])).unsqueeze(1).to(self.device)
            gd_targets_pi = torch.FloatTensor(np.array([s[1] for s in gd_batch])).to(self.device)

            # Hook to capture backbone output tensor (with grad tracking)
            bb_ref = {}
            def hook_bb_capture(module, inp, out):
                bb_ref['x'] = inp[0]
            h_bb = self.net.value_conv.register_forward_hook(hook_bb_capture)

            self.optimizer.zero_grad()
            gd_pred_v, gd_pred_p = self.net(gd_states)

            gd_v_loss = F.mse_loss(gd_pred_v, gd_targets_v)
            gd_p_loss = -torch.mean(torch.sum(gd_targets_pi * torch.log(gd_pred_p + 1e-8), dim=1))

            bb_x = bb_ref['x']  # [batch, 256, H, W]
            h_bb.remove()

            # (A) Per-channel gradient from each head on backbone output
            v_grad_bb = torch.autograd.grad(gd_v_loss, bb_x, retain_graph=True)[0]
            p_grad_bb = torch.autograd.grad(gd_p_loss, bb_x, retain_graph=True)[0]

            v_grad_ch = v_grad_bb.abs().mean(dim=(0, 2, 3)).cpu().numpy()  # [channels]
            p_grad_ch = p_grad_bb.abs().mean(dim=(0, 2, 3)).cpu().numpy()  # [channels]

            v_grad_bb_norm = float(v_grad_bb.norm().item())
            p_grad_bb_norm = float(p_grad_bb.norm().item())

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
            p_bp_grads = torch.autograd.grad(gd_p_loss, bp_list, allow_unused=True)

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
            })
        except Exception:
            pass

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
                    all_rb_bn[bi] = {
                        "dead": rb_bn_dead,
                        "neg_gamma": rb_neg_gamma,
                        "eff_gain_mean": float(np.abs(rb_eff_gain_np).mean()),
                        "eff_gain_max": float(np.abs(rb_eff_gain_np).max()),
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
            })
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
                for sym_input, sym_policy in self.game.get_symmetries(ex[0], ex[1]):
                    augmented.append([sym_input, sym_policy, ex[2]])
            self.buffer.insert_batch(augmented)

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

            # Train
            t0 = time.time()
            train_result = self.train_network()
            train_time = time.time() - t0

            iter_time = time.time() - iter_t0

            if train_result is not None:
                avg_loss, avg_value_loss, avg_policy_loss = train_result
                self.writer.add_scalar("loss/total", avg_loss, iteration)
                self.writer.add_scalar("loss/value", avg_value_loss, iteration)
                self.writer.add_scalar("loss/policy", avg_policy_loss, iteration)
                print(f"  Iter {iteration+1}/{num_iterations}: loss={avg_loss:.4f} "
                      f"(v={avg_value_loss:.4f} p={avg_policy_loss:.4f}) | "
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
                # Console
                print(f"  Diag: targets mean={d['val_target_mean']:+.3f} "
                      f"std={d['val_target_std']:.3f} | "
                      f"X={d['frac_neg']:.1%} O={d['frac_pos']:.1%} "
                      f"draw={d['frac_draw']:.1%}")
                overfit_gap = d['val_vloss'] - d['late_vloss']
                print(f"  Diag: eff_epochs={d['effective_epochs']:.1f} "
                      f"vlw={d.get('effective_vlw',1.0):.2f} "
                      f"steps={d['num_steps']} | "
                      f"vloss train={d['late_vloss']:.4f} "
                      f"val={d['val_vloss']:.4f} "
                      f"(gap={overfit_gap:+.4f}) | "
                      f"buf={d['buffer_fill']}/{d['buffer_capacity']}"
                      f"{' FULL' if d['buffer_full'] else ''}")
                self.writer.add_scalar("diag/val_vloss", d["val_vloss"], iteration)
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
                # (C) Value confidence calibration
                print(f"  Diag[C]: confident_acc={d['value_confidence_acc']:.1%} "
                      f"(frac_confident={d['value_confident_frac']:.1%})")
                # (RB) Per-block gradient norms
                rb_gn = d.get('rb_grad_norms', {})
                if rb_gn:
                    rb_str = " ".join(f"rb{i}={n:.4f}" for i, n in sorted(rb_gn.items()))
                    print(f"  Diag[RB]: grad_norms: {rb_str}")
                # Value target histogram
                vh_bins = d.get('val_hist', [])
                if vh_bins:
                    print(f"  Diag[TH]: targets [-1,-0.5)={vh_bins[0]:.1%} "
                          f"[-0.5,0)={vh_bins[1]:.1%} [0]={vh_bins[2]:.1%} "
                          f"(0,0.5]={vh_bins[3]:.1%} (0.5,1]={vh_bins[4]:.1%}")
                # (A) Per-player value breakdown
                print(f"  Diag[A]: X_vloss={d['x_vloss']:.4f} O_vloss={d['o_vloss']:.4f} | "
                      f"X_target={d['x_target_mean']:+.3f} O_target={d['o_target_mean']:+.3f} | "
                      f"X_pred={d['x_pred_mean']:+.3f} O_pred={d['o_pred_mean']:+.3f}")
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
                          f"pre_tanh=[{vh['pre_tanh_min']:+.2f},{vh['pre_tanh_max']:+.2f}] "
                          f"std={vh['pre_tanh_std']:.3f} "
                          f"saturated={vh['out_saturated']:.1%} "
                          f"|v|={vh['out_abs_mean']:.3f}")
                    print(f"  Diag[V2]: fc2_w=[{vh['fc2_w_min']:+.3f},{vh['fc2_w_max']:+.3f}] "
                          f"norm={vh['fc2_w_norm']:.3f} bias={vh['fc2_bias']:+.4f} | "
                          f"fc1_w_norm={vh['fc1_w_norm']:.3f} | "
                          f"backbone std={vh['backbone_std']:.3f} |x|={vh['backbone_abs_mean']:.3f}")
                    print(f"  Diag[V3]: spike_neuron={vh['spike_neuron']} "
                          f"contrib={vh['spike_contribution']:+.3f} "
                          f"(fc2_w={vh['spike_fc2_weight']:+.4f} * "
                          f"act={vh['spike_fc1_act']:+.4f})")
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
                                         f"{rbd.get('eff_gain_max',0):.2f}")
                            parts.append(f"svd_rank90={rbd.get('svd_rank90',0)}/"
                                         f"{rbd.get('svd_total',0)}")
                        if bi in rb_act_data:
                            rad = rb_act_data[bi]
                            parts.append(f"|x|={rad['abs_mean']:.3f} "
                                         f"std={rad['std']:.3f} "
                                         f"dead={rad['dead_channels']}")
                        print(" | ".join(parts))
                    # Tensorboard
                    self.writer.add_scalar("vh/dead_neurons", vh["dead_neurons"], iteration)
                    self.writer.add_scalar("vh/active_neurons", vh.get("active_neurons", 0), iteration)
                    self.writer.add_scalar("vh/pre_tanh_max", vh["pre_tanh_max"], iteration)
                    self.writer.add_scalar("vh/pre_tanh_min", vh["pre_tanh_min"], iteration)
                    self.writer.add_scalar("vh/saturated", vh["out_saturated"], iteration)
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
                    for bi, rad in vh.get('rb_act_stats', {}).items():
                        self.writer.add_scalar(f"rb{bi}/act_abs_mean", rad["abs_mean"], iteration)
                        self.writer.add_scalar(f"rb{bi}/act_std", rad["std"], iteration)
                        self.writer.add_scalar(f"rb{bi}/act_dead_channels", rad["dead_channels"], iteration)

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
                          f"saturated={vd['frac_saturated_any']:.1%} | "
                          f"sign_acc={vd['sign_accuracy']:.1%} | "
                          f"MAE={vd['mae_vs_outcome']:.3f} | "
                          f"corr={vd['pred_outcome_corr']:+.3f}")
                    print(f"  SelfPlay: v_when_X={vd['mean_when_x_moves']:+.3f} "
                          f"v_when_O={vd['mean_when_o_moves']:+.3f} | "
                          f"sat+={vd['frac_saturated_pos']:.1%} "
                          f"sat-={vd['frac_saturated_neg']:.1%}")
                    if 'mcts_visit_entropy_mean' in vd:
                        self.writer.add_scalar("selfplay_diag/mcts_visit_entropy",
                                               vd["mcts_visit_entropy_mean"], iteration)
                        print(f"  SelfPlay: mcts_visit_entropy="
                              f"{vd['mcts_visit_entropy_mean']:.3f} "
                              f"(std={vd['mcts_visit_entropy_std']:.3f})")

            # === Fixed diagnostic position evaluation ===
            self._eval_diagnostic_positions(iteration)

            # Save every 10 iterations + always on the last one
            # Also save iteration 0 if no checkpoint exists (quick sanity check)
            no_checkpoint = not os.path.exists(os.path.join(self.checkpoint_dir, "latest.txt"))
            if (iteration + 1) % 10 == 0 or iteration == num_iterations - 1 or (iteration == 0 and no_checkpoint):
                self.net.save(self.checkpoint_dir)

        self.writer.close()

    def _eval_diagnostic_positions(self, iteration):
        """Evaluate the network on fixed diagnostic positions every iteration.

        This helps track how the value head evolves over training on known positions.
        """
        self.net.eval()
        positions = self._get_diagnostic_positions()
        if not positions:
            return

        print(f"  FixedEval:")
        for name, state_input, expected_str in positions:
            value, policy = self.net.predict(state_input)
            top_action = np.argmax(policy)
            self.writer.add_scalar(f"fixed_eval/{name}_value", value, iteration)
            self.writer.add_scalar(f"fixed_eval/{name}_top_action", top_action, iteration)
            print(f"    {name}: V={value:+.4f} top_act={top_action} ({expected_str})")

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
        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = -1  # X has 3 in row
        board[0][4] = 1     # some O piece
        s = C4State(None, board, player=-1)
        positions.append(("x_wins_next", self.game.state_to_input(s), "expect > +0.5 (I'm winning)"))

        # Position 3: O about to win horizontally (O to move)
        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = 1   # O has 3 in row
        board[0][4] = -1    # some X piece
        board[1][4] = -1
        s = C4State(None, board, player=1)
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
