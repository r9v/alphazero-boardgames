import random
import numpy as np
import torch
import torch.nn.functional as F

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
            print(f"  Not enough samples ({len(samples)}), skipping training")
            return

        self.net.train()
        total_loss = 0
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
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Training: {len(samples)} samples, {num_batches} batches, avg loss: {avg_loss:.4f}")

    def run(self, num_iterations=1):
        """Run the training loop: self-play → train → save."""
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            for game_num in range(self.games_per_iteration):
                examples = self.self_play_game()
                self.buffer.insert_batch(examples)
                result = examples[-1][2] if examples else 0
                print(f"  Game {game_num + 1}/{self.games_per_iteration} — result: {result}")
            self.train_network()
            self.net.save(self.checkpoint_dir)
