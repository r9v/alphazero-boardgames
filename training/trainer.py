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
from training.training_diagnostics import (
    make_accumulator, collect_step_diagnostics, aggregate_training_results,
)
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
        self.logger = TrainingLogger(self.writer)

    def train_network(self, n_new_positions=0):
        """Train the network on samples from the replay buffer."""
        samples = [s for s in self.buffer.arr if s is not None]
        if len(samples) < self.batch_size:
            print(f"  Not enough samples ({len(samples)}), skipping training")
            return None

        # Snapshot backbone params for drift measurement
        with torch.no_grad():
            pre_train_backbone = torch.cat([
                p.data.flatten() for p in self.net.res_blocks.parameters()
            ]).clone()

        setup = self._init_training_state(samples, n_new_positions)
        samples = setup['train_samples']
        acc = setup['acc']
        cfg = {k: setup[k] for k in ('num_steps', 'effective_vlw', 'effective_epochs',
                                       'early_cutoff', 'late_start', 'lr_min')}

        grad_stats_list = []

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

            collect_step_diagnostics(
                step, states, target_vs, target_pis,
                pred_vs, pred_pi_logits, value_loss, policy_loss,
                acc, cfg, self.net, self._value_params, self._policy_params,
                grad_stats_list, self.device)

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
        train_losses, train_diag, train_perf = aggregate_training_results(
            acc, setup['val_samples'], cfg, setup,
            self.net, self.batch_size, self.device, self.use_amp,
            len(self.buffer), self.buffer.max_size,
            pre_train_backbone, grad_stats_list)

        # Store for any external consumers that read these attributes
        self._train_diag = train_diag
        self._train_perf = train_perf

        return train_losses

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

        acc = make_accumulator()

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

            # Build self-play value diagnostics dict for the logger
            value_diag = {}
            if hasattr(self, '_batched') and hasattr(self._batched, 'value_diag'):
                value_diag = self._batched.value_diag or {}

            train_diag = self._train_diag if hasattr(self, '_train_diag') else {}

            self.logger.log_iteration(iteration, num_iterations, {
                'train_result': train_result,
                'wins_p1': wins_p1, 'wins_p2': wins_p2, 'draws': draws,
                'avg_length': avg_length, 'min_length': min_length,
                'max_length': max_length, 'p1_win_pct': p1_win_pct,
                'self_play_time': self_play_time, 'train_time': train_time,
                'iter_time': iter_time,
                'train_diag': train_diag,
                'value_diag': value_diag,
            })

            # Save every 5 iterations + always on the last one
            # Also save iteration 0 if no checkpoint exists (quick sanity check)
            no_checkpoint = not os.path.exists(os.path.join(self.checkpoint_dir, "latest.txt"))
            if (iteration + 1) % 5 == 0 or iteration == num_iterations - 1 or (iteration == 0 and no_checkpoint):
                self.net.save(self.checkpoint_dir, iteration=iteration, num_iterations=num_iterations)

        self.logger.close()
