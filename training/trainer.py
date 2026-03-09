import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
        self.max_train_steps = self.config.get("max_train_steps", 1000)
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
            tqdm.write(f"  Not enough samples ({len(samples)}), skipping training")
            return None

        self.net.train()
        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        num_batches = 0
        data_prep_time = 0.0
        gradient_time = 0.0

        # Use fixed number of training steps with random sampling
        n_samples = len(samples)
        num_steps = min(self.max_train_steps, self.epochs * (n_samples // self.batch_size))

        for _ in range(num_steps):
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
            loss = value_loss + policy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            gradient_time += time.time() - t0

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        self._train_perf = {
            "data_prep_time": data_prep_time,
            "gradient_time": gradient_time,
            "num_samples": len(samples),
            "num_batches": num_batches,
        }
        return avg_loss, avg_value_loss, avg_policy_loss

    def _self_play(self, iteration):
        """Run self-play games in parallel with batched evaluation."""
        self._batched = BatchedSelfPlay(
            self.game, self.net, self.games_per_iteration, self.num_simulations
        )
        return self._batched.play_games()

    def run(self, num_iterations=1):
        """Run the training loop: self-play → train → save."""
        for iteration in tqdm(range(num_iterations), desc="Iterations", unit="iter"):
            # Self-play
            t0 = time.time()
            all_examples, results, game_lengths = self._self_play(iteration)
            self_play_time = time.time() - t0

            self.buffer.insert_batch(all_examples)

            # Log self-play stats
            wins_p1 = results.count(-1)
            wins_p2 = results.count(1)
            draws = results.count(0)
            avg_length = np.mean(game_lengths)
            self.writer.add_scalar("self_play/avg_game_length", avg_length, iteration)
            self.writer.add_scalar("self_play/wins_p1", wins_p1, iteration)
            self.writer.add_scalar("self_play/wins_p2", wins_p2, iteration)
            self.writer.add_scalar("self_play/draws", draws, iteration)
            self.writer.add_scalar("self_play/buffer_size",
                                   sum(1 for s in self.buffer.arr if s is not None), iteration)

            # Train
            t0 = time.time()
            train_result = self.train_network()
            train_time = time.time() - t0

            if train_result is not None:
                avg_loss, avg_value_loss, avg_policy_loss = train_result
                self.writer.add_scalar("loss/total", avg_loss, iteration)
                self.writer.add_scalar("loss/value", avg_value_loss, iteration)
                self.writer.add_scalar("loss/policy", avg_policy_loss, iteration)
                tqdm.write(f"  Iter {iteration+1}: loss={avg_loss:.4f} "
                           f"(v={avg_value_loss:.4f} p={avg_policy_loss:.4f}) | "
                           f"games: p1={wins_p1} p2={wins_p2} draw={draws} | "
                           f"avg_len={avg_length:.1f} | "
                           f"self_play={self_play_time:.1f}s train={train_time:.1f}s")

            self.writer.add_scalar("perf/self_play_time", self_play_time, iteration)
            self.writer.add_scalar("perf/train_time", train_time, iteration)
            if hasattr(self, '_batched') and hasattr(self._batched, 'perf'):
                perf = self._batched.perf
                mcts_time = perf["select_expand_time"] + perf["backup_time"]
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
                tqdm.write(f"  MCTS: select={perf['select_expand_time']:.1f}s "
                           f"backup={perf['backup_time']:.1f}s "
                           f"terminal_hits={perf['terminal_hits']}")
                tqdm.write(f"  NN:   forward={perf['forward_time']:.1f}s "
                           f"result={perf['result_time']:.1f}s "
                           f"preprocess={perf['preprocess_time']:.1f}s "
                           f"transfer={perf['transfer_time']:.1f}s | "
                           f"batches={perf['batch_count']} "
                           f"batch_sz={perf['min_batch']}/{avg_batch:.0f}/{perf['max_batch']}")
            if hasattr(self, '_train_perf'):
                tp = self._train_perf
                self.writer.add_scalar("perf/train_data_prep", tp["data_prep_time"], iteration)
                self.writer.add_scalar("perf/train_gradient", tp["gradient_time"], iteration)
                self.writer.add_scalar("perf/train_num_samples", tp["num_samples"], iteration)
                self.writer.add_scalar("perf/train_num_batches", tp["num_batches"], iteration)
                tqdm.write(f"  Train: data={tp['data_prep_time']:.1f}s "
                           f"grad={tp['gradient_time']:.1f}s | "
                           f"samples={tp['num_samples']} "
                           f"batches={tp['num_batches']}")

            # Save every 15 iterations + always on the last one
            if (iteration + 1) % 15 == 0 or iteration == num_iterations - 1:
                self.net.save(self.checkpoint_dir)

        self.writer.close()
