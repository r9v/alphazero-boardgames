import time
import numpy as np

from mcts.mcts import MCTS, Node, add_dirichlet_noise


class BatchedSelfPlay:
    """Runs N self-play games in parallel, batching neural network evaluations.

    Instead of evaluating one position at a time (slow on GPU due to kernel
    launch overhead), this collects pending MCTS leaf nodes across all games
    and evaluates them in a single batched forward pass.
    """

    def __init__(self, game, net, num_games, num_simulations):
        self.game = game
        self.net = net
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.mcts = MCTS(game, net)

    def play_games(self):
        """Play num_games self-play games in parallel.

        Returns:
            (all_examples, results, game_lengths) where:
            - all_examples: flat list of [state_input, policy, outcome]
            - results: list of terminal_value per game
            - game_lengths: list of number of moves per game
        """
        self._last_select_expand_time = 0.0
        self._last_backup_time = 0.0
        self._last_nn_time = 0.0
        self._last_preprocess_time = 0.0
        self._last_transfer_time = 0.0
        self._last_forward_time = 0.0
        self._last_postprocess_time = 0.0
        self._last_batch_count = 0
        self._last_sample_count = 0
        self._last_terminal_hits = 0
        self._last_min_batch = float('inf')
        self._last_max_batch = 0

        # Initialize all games
        states = [self.game.new_game() for _ in range(self.num_games)]
        examples = [[] for _ in range(self.num_games)]
        active = list(range(self.num_games))  # indices of games still in progress

        # Create initial roots (deferred — no net eval yet)
        roots = [Node(None, s, self.game) for s in states]

        # Batch-evaluate all initial roots
        self._batch_evaluate_nodes(roots)

        # Add Dirichlet noise to root priors
        for root in roots:
            root.P = add_dirichlet_noise(root.P, 0.03, 0.25)

        while active:
            # Run MCTS simulations for all active games
            self._run_simulations(roots, active)

            # Extract policies and pick moves
            next_active = []
            for i in active:
                root = roots[i]
                pi = np.zeros(np.shape(root.available_actions_mask))
                for action in root.available_actions:
                    if root.children[action] is not None:
                        pi[action] = root.children[action].n / root.n

                action = np.random.choice(len(pi), p=pi)
                examples[i].append([self.game.state_to_input(states[i]), pi])

                states[i] = self.game.step(states[i], action)

                if states[i].terminal:
                    # Game over — attach terminal value to all examples
                    for ex in examples[i]:
                        ex.append(states[i].terminal_value)
                else:
                    # Create new root for next move (deferred)
                    roots[i] = Node(None, states[i], self.game)
                    next_active.append(i)

            # Batch-evaluate new roots for games that are still active
            if next_active:
                new_roots = [roots[i] for i in next_active]
                self._batch_evaluate_nodes(new_roots)
                for i in next_active:
                    roots[i].P = add_dirichlet_noise(roots[i].P, 0.03, 0.25)

            active = next_active

        # Collect stats and flatten examples
        all_examples = []
        results = []
        game_lengths = []
        for game_examples in examples:
            all_examples.extend(game_examples)
            game_lengths.append(len(game_examples))
            results.append(game_examples[-1][2] if game_examples else 0)

        self.perf = {
            "select_expand_time": self._last_select_expand_time,
            "backup_time": self._last_backup_time,
            "nn_time": self._last_nn_time,
            "preprocess_time": self._last_preprocess_time,
            "transfer_time": self._last_transfer_time,
            "forward_time": self._last_forward_time,
            "postprocess_time": self._last_postprocess_time,
            "batch_count": self._last_batch_count,
            "sample_count": self._last_sample_count,
            "terminal_hits": self._last_terminal_hits,
            "min_batch": self._last_min_batch if self._last_min_batch != float('inf') else 0,
            "max_batch": self._last_max_batch,
        }
        return all_examples, results, game_lengths

    def _run_simulations(self, roots, active):
        """Run num_simulations MCTS simulations for all active games.

        Each round expands one node per game, batching them for the GPU.
        """
        select_expand_time = 0.0
        backup_time = 0.0
        nn_time = 0.0
        terminal_hits = 0

        for _ in range(self.num_simulations):
            pending = []

            t0 = time.time()
            for i in active:
                leaf = self.mcts.search_expand(roots[i])
                if leaf is not None:
                    pending.append(leaf)
                else:
                    terminal_hits += 1
            select_expand_time += time.time() - t0

            if pending:
                t0 = time.time()
                self._batch_evaluate_nodes(pending)
                nn_time += time.time() - t0

                t0 = time.time()
                for node in pending:
                    self.mcts.search_backup(node)
                backup_time += time.time() - t0

        self._last_select_expand_time += select_expand_time
        self._last_backup_time += backup_time
        self._last_nn_time = getattr(self, '_last_nn_time', 0) + nn_time
        self._last_terminal_hits += terminal_hits

    def _batch_evaluate_nodes(self, nodes):
        """Batch neural network evaluation for a list of nodes."""
        if not nodes:
            return

        t0 = time.time()
        state_inputs = [self.game.state_to_input(node.state) for node in nodes]
        preprocess_time = time.time() - t0

        values, policies, detail = self.net.batch_predict(state_inputs, detailed_timing=True)

        t0 = time.time()
        for node, value, policy in zip(nodes, values, policies):
            node.resolve(value, policy)
        postprocess_time = time.time() - t0

        self._last_preprocess_time += preprocess_time
        self._last_transfer_time += detail["transfer_time"]
        self._last_forward_time += detail["forward_time"]
        self._last_postprocess_time += postprocess_time
        self._last_batch_count += 1
        self._last_sample_count += len(nodes)
        self._last_min_batch = min(self._last_min_batch, len(nodes))
        self._last_max_batch = max(self._last_max_batch, len(nodes))
