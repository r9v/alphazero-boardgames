import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mcts import MCTS
from training.replay_buffer import ReplayBuffer


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
        self.buffer = ReplayBuffer(self.config.get("buffer_size", 100000))
        self.mcts = MCTS(game, net)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)

        log_dir = self.config.get("log_dir", f"runs/{self.config.get('game_name', 'unknown')}")
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

    def self_play_game(self):
        """Play a single self-play game and return training examples."""
        state = self.game.new_game()
        examples = []
        while True:
            pi = self.mcts.get_policy(self.num_simulations, state, add_dirichlet=True)
            action = np.random.choice(len(pi), p=pi)
            examples.append([self.game.state_to_input(state), pi])
            state = self.game.step(state, action)
            if state.terminal:
                for ex in examples:
                    ex.append(state.terminal_value)
                return examples

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

        for epoch in range(self.epochs):
            random.shuffle(samples)
            for i in range(0, len(samples) - self.batch_size + 1, self.batch_size):
                batch = samples[i:i + self.batch_size]

                states = torch.FloatTensor(np.array([s[0] for s in batch]))
                target_pis = torch.FloatTensor(np.array([s[1] for s in batch]))
                target_vs = torch.FloatTensor(np.array([s[2] for s in batch])).unsqueeze(1)

                pred_vs, pred_pis = self.net(states)

                value_loss = F.mse_loss(pred_vs, target_vs)
                policy_loss = -torch.mean(torch.sum(target_pis * torch.log(pred_pis + 1e-8), dim=1))
                loss = value_loss + policy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        return avg_loss, avg_value_loss, avg_policy_loss

    def run(self, num_iterations=1):
        """Run the training loop: self-play → train → save."""
        for iteration in tqdm(range(num_iterations), desc="Iterations", unit="iter"):
            # Self-play
            results = []
            game_lengths = []
            for game_num in tqdm(range(self.games_per_iteration),
                                 desc=f"  Self-play (iter {iteration+1})",
                                 unit="game", leave=False):
                examples = self.self_play_game()
                self.buffer.insert_batch(examples)
                result = examples[-1][2] if examples else 0
                results.append(result)
                game_lengths.append(len(examples))

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
            train_result = self.train_network()
            if train_result is not None:
                avg_loss, avg_value_loss, avg_policy_loss = train_result
                self.writer.add_scalar("loss/total", avg_loss, iteration)
                self.writer.add_scalar("loss/value", avg_value_loss, iteration)
                self.writer.add_scalar("loss/policy", avg_policy_loss, iteration)
                tqdm.write(f"  Iter {iteration+1}: loss={avg_loss:.4f} "
                           f"(v={avg_value_loss:.4f} p={avg_policy_loss:.4f}) | "
                           f"games: p1={wins_p1} p2={wins_p2} draw={draws} | "
                           f"avg_len={avg_length:.1f}")

            self.net.save(self.checkpoint_dir)

        self.writer.close()
