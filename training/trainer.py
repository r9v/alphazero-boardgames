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
        self.value_loss_weight = self.config.get("value_loss_weight", 1.0)
        self.buffer = ReplayBuffer(self.config.get("buffer_size", 100000))
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

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

        # Dynamic training steps: target N effective epochs, capped by max_train_steps
        # Scale epochs with buffer fill to prevent memorization when buffer is small
        n_samples = len(samples)
        buffer_capacity = self.buffer.max_size
        fill_ratio = min(n_samples / max(buffer_capacity, 1), 1.0)
        # Linearly scale from 1 epoch (empty) to target_epochs (full)
        scaled_epochs = 1.0 + (self.target_epochs - 1.0) * fill_ratio
        target_steps = int(scaled_epochs * (n_samples // self.batch_size))
        num_steps = max(1, min(self.max_train_steps, target_steps))
        effective_epochs = (num_steps * self.batch_size) / n_samples
        early_cutoff = max(num_steps // 10, 1)
        late_start = num_steps - early_cutoff

        # Cosine LR schedule: lr decays from initial to 10% over training steps
        import math
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
            loss = self.value_loss_weight * value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()

            # (A) Per-player loss breakdown every 10 steps
            if step % 10 == 0:
                with torch.no_grad():
                    # Infer player from piece counts: equal pieces = X to move
                    num_hist = getattr(self.game, 'num_history_states', 2)
                    ch = 2 * num_hist
                    my_counts = states[:, ch].sum(dim=(1, 2))
                    opp_counts = states[:, ch + 1].sum(dim=(1, 2))
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

        # (V) Value head health: dead neurons, pre-tanh range, saturation
        vh_diag = {}
        try:
            self.net.eval()
            with torch.no_grad():
                # Use a sample of training data
                diag_batch = random.choices(samples, k=min(256, len(samples)))
                diag_inp = torch.FloatTensor(
                    np.array([s[0] for s in diag_batch])
                ).to(self.device)

                # Run through backbone
                x_bb = F.relu(self.net.bn(self.net.conv(diag_inp)))
                for block in self.net.res_blocks:
                    x_bb = block(x_bb)

                # Value head layers
                v_conv = F.relu(self.net.value_bn(self.net.value_conv(x_bb)))
                v_flat = v_conv.view(v_conv.size(0), -1)
                v_fc1_out = F.relu(self.net.value_fc1(v_flat))
                v_pre_tanh = self.net.value_fc2(v_fc1_out)
                v_out = torch.tanh(v_pre_tanh)

                fc1_np = v_fc1_out.cpu().numpy()
                pre_tanh_np = v_pre_tanh.cpu().numpy().flatten()
                out_np = v_out.cpu().numpy().flatten()

                n_total_neurons = fc1_np.shape[1]
                dead_mask = (fc1_np == 0).all(axis=0)
                n_dead = int(dead_mask.sum())

                vh_diag = {
                    "dead_neurons": n_dead,
                    "total_neurons": n_total_neurons,
                    "pre_tanh_min": float(pre_tanh_np.min()),
                    "pre_tanh_max": float(pre_tanh_np.max()),
                    "pre_tanh_std": float(pre_tanh_np.std()),
                    "out_saturated": float((np.abs(out_np) > 0.95).mean()),
                    "out_abs_mean": float(np.abs(out_np).mean()),
                }
            self.net.train()
        except Exception:
            pass

        # (F) Gradient stats summary
        grad_stats_summary = {}
        if hasattr(self, '_grad_stats') and self._grad_stats:
            gs = self._grad_stats
            grad_stats_summary = {
                'fc1_grad_norm_mean': np.mean([g['fc1_grad_norm'] for g in gs]),
                'fc2_grad_norm_mean': np.mean([g['fc2_grad_norm'] for g in gs]),
                'fc1_grad_mean': np.mean([g['fc1_grad_mean'] for g in gs]),
                'fc2_grad_mean': np.mean([g['fc2_grad_mean'] for g in gs]),
                'error_mean_trend': [g['error_mean'] for g in gs],
            }
            self._grad_stats = []

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
        }
        return avg_loss, avg_value_loss, avg_policy_loss

    def _self_play(self, iteration):
        """Run self-play games in parallel with batched evaluation."""
        # Log first 2 games in detail every 5 iterations (or first iteration)
        log_games = 2 if (iteration % 5 == 0 or iteration == 0) else 0
        self._batched = BatchedSelfPlay(
            self.game, self.net, self.games_per_iteration, self.num_simulations,
            selects_per_round=self.config.get("selects_per_round", 1),
            vl_value=self.config.get("vl_value", 0.0),
            log_games=log_games,
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
                # Console
                print(f"  Diag: targets mean={d['val_target_mean']:+.3f} "
                      f"std={d['val_target_std']:.3f} | "
                      f"X={d['frac_neg']:.1%} O={d['frac_pos']:.1%} "
                      f"draw={d['frac_draw']:.1%}")
                overfit_gap = d['val_vloss'] - d['late_vloss']
                print(f"  Diag: eff_epochs={d['effective_epochs']:.1f} "
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
                    print(f"  Diag[V]: dead_neurons={vh['dead_neurons']}/{vh['total_neurons']} "
                          f"pre_tanh=[{vh['pre_tanh_min']:+.2f},{vh['pre_tanh_max']:+.2f}] "
                          f"std={vh['pre_tanh_std']:.3f} "
                          f"saturated={vh['out_saturated']:.1%} "
                          f"|v|={vh['out_abs_mean']:.3f}")
                    self.writer.add_scalar("vh/dead_neurons", vh["dead_neurons"], iteration)
                    self.writer.add_scalar("vh/pre_tanh_max", vh["pre_tanh_max"], iteration)
                    self.writer.add_scalar("vh/pre_tanh_min", vh["pre_tanh_min"], iteration)
                    self.writer.add_scalar("vh/saturated", vh["out_saturated"], iteration)

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

            # === Log detailed game trace (first logged game) ===
            if hasattr(self, '_batched') and self._batched._game_logs:
                for gi, glog in enumerate(self._batched._game_logs):
                    if not glog:
                        continue
                    if gi >= 1:  # only print first logged game
                        break
                    outcome = results[gi]
                    winner = "X" if outcome == -1 else ("O" if outcome == 1 else "draw")
                    print(f"  GameTrace[{gi}]: {len(glog)} moves, winner={winner} (val={outcome})")
                    for entry in glog:
                        player_str = "X" if entry["player"] == -1 else "O"
                        pi = entry["pi"]
                        best_action = max(range(len(pi)), key=lambda a: pi[a])
                        # Show NN value vs eventual outcome
                        nnet_v = entry["nnet_value"]
                        # Relative: target from this player's perspective
                        target = outcome * entry["player"]
                        err = abs(nnet_v - target)
                        Qs_str = " ".join(f"{a}:{q:+.2f}" for a, q in sorted(entry["child_Qs"].items()))
                        Ns_str = " ".join(f"{a}:{n}" for a, n in sorted(entry["child_Ns"].items()))
                        print(f"    mv{entry['move']:>2} {player_str}: "
                              f"V={nnet_v:+.3f} (err={err:.2f}) "
                              f"act={best_action} | "
                              f"Q=[{Qs_str}] N=[{Ns_str}]")

            # === Fixed diagnostic position evaluation ===
            self._eval_diagnostic_positions(iteration)

            # Save every 15 iterations + always on the last one
            # Also save iteration 0 if no checkpoint exists (quick sanity check)
            no_checkpoint = not os.path.exists(os.path.join(self.checkpoint_dir, "latest.txt"))
            if (iteration + 1) % 15 == 0 or iteration == num_iterations - 1 or (iteration == 0 and no_checkpoint):
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
