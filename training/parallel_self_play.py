import time
import numpy as np

try:
    from mcts.c_mcts import CMCTS as MCTS, CNode as Node
    from mcts.mcts import add_dirichlet_noise
except ImportError:
    from mcts.mcts import MCTS, Node, add_dirichlet_noise


class BatchedSelfPlay:
    """Runs N self-play games in parallel, batching neural network evaluations.

    Instead of evaluating one position at a time (slow on GPU due to kernel
    launch overhead), this collects pending MCTS leaf nodes across all games
    and evaluates them in a single batched forward pass.
    """

    def __init__(self, game, net, num_games, num_simulations,
                 selects_per_round=1, vl_value=0.0, log_games=0,
                 temp_threshold=15, c_puct=1.5):
        self.game = game
        self.net = net
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.selects_per_round = selects_per_round
        self.vl_value = vl_value
        self.temp_threshold = temp_threshold
        self.mcts = MCTS(game, net, c_puct=c_puct)
        self.log_games = log_games  # how many games to log in detail

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
        self._last_result_time = 0.0
        self._last_postprocess_time = 0.0
        self._last_batch_count = 0
        self._last_sample_count = 0
        self._last_terminal_hits = 0
        self._last_min_batch = float('inf')
        self._last_max_batch = 0
        self._encoding_checks = 0
        self._encoding_errors = 0

        # Diagnostic: per-move logs for a sample of games
        # Each entry: {move, player, nnet_value, action, pi, child_Qs, child_Ns}
        self._game_logs = [[] for _ in range(self.num_games)]
        self._game_value_preds = [[] for _ in range(self.num_games)]  # (player, nnet_value) per move

        # Initialize all games
        states = [self.game.new_game() for _ in range(self.num_games)]
        examples = [[] for _ in range(self.num_games)]
        terminal_values = [0] * self.num_games  # raw terminal values per game
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
                children = root.children
                for action in root.available_actions:
                    child = children.get(action) if isinstance(children, dict) else children[action]
                    if child is not None:
                        pi[action] = child.n / root.n

                # Diagnostic: record per-move stats
                self._game_value_preds[i].append((states[i].player, root.nnet_value))
                if i < self.log_games:
                    child_Qs = {}
                    child_Ns = {}
                    for a in root.available_actions:
                        ch = children.get(a) if isinstance(children, dict) else children[a]
                        if ch is not None:
                            child_Qs[int(a)] = ch.Q
                            child_Ns[int(a)] = ch.n
                    self._game_logs[i].append({
                        "move": len(examples[i]),
                        "player": states[i].player,
                        "nnet_value": root.nnet_value,
                        "root_Q": root.Q,
                        "root_N": root.n,
                        "pi": pi.tolist(),
                        "child_Qs": child_Qs,
                        "child_Ns": child_Ns,
                    })

                # Temperature: explore early, exploit late
                move_num = len(examples[i])
                if move_num < self.temp_threshold:
                    action = np.random.choice(len(pi), p=pi)
                else:
                    action = np.argmax(pi)
                # Training target is always the full visit distribution
                examples[i].append([self.game.state_to_input(states[i]), pi, states[i].player])

                states[i] = self.game.step(states[i], action)

                if states[i].terminal:
                    # Game over — store raw terminal value and compute targets
                    tv = states[i].terminal_value
                    terminal_values[i] = tv
                    for ex in examples[i]:
                        player_at_pos = ex[2]
                        # Relative: target from current player's perspective
                        target = tv * player_at_pos
                        # Assertion: winner's positions should get +1, loser's -1
                        if tv != 0:  # not a draw
                            assert target in (-1.0, 1.0), \
                                f"Bad target {target}: tv={tv}, player={player_at_pos}"
                        ex[2] = target
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
        results = []  # raw terminal values: -1=X wins, +1=O wins, 0=draw
        game_lengths = []
        for i, game_examples in enumerate(examples):
            all_examples.extend(game_examples)
            game_lengths.append(len(game_examples))
            results.append(terminal_values[i])

        # Verify results contain raw terminal values, not transformed targets
        n_x_wins = results.count(-1)
        n_o_wins = results.count(1)
        n_draws = results.count(0)
        assert n_x_wins + n_o_wins + n_draws == len(results), \
            f"Results contain unexpected values: {set(results)}"

        self.perf = {
            "select_expand_time": self._last_select_expand_time,
            "backup_time": self._last_backup_time,
            "nn_time": self._last_nn_time,
            "preprocess_time": self._last_preprocess_time,
            "transfer_time": self._last_transfer_time,
            "forward_time": self._last_forward_time,
            "result_time": self._last_result_time,
            "postprocess_time": self._last_postprocess_time,
            "batch_count": self._last_batch_count,
            "sample_count": self._last_sample_count,
            "terminal_hits": self._last_terminal_hits,
            "min_batch": self._last_min_batch if self._last_min_batch != float('inf') else 0,
            "max_batch": self._last_max_batch,
            "encoding_checks": self._encoding_checks,
            "encoding_errors": self._encoding_errors,
        }

        # Compute self-play value prediction diagnostics
        self._compute_value_diagnostics(results)

        return all_examples, results, game_lengths

    def _compute_value_diagnostics(self, results):
        """Compute statistics about NN value predictions during self-play."""
        all_preds = []   # (nnet_value, target_outcome, player)
        for i, game_preds in enumerate(self._game_value_preds):
            outcome = results[i]
            for player, nnet_v in game_preds:
                # Relative: compare nnet against target from player's perspective
                target = outcome * player
                all_preds.append((nnet_v, target, player))

        if not all_preds:
            self.value_diag = {}
            return

        nnet_vals = np.array([p[0] for p in all_preds])
        targets = np.array([p[1] for p in all_preds])
        players = np.array([p[2] for p in all_preds])

        # Value prediction distribution
        self.value_diag = {
            "mean_nnet_value": float(nnet_vals.mean()),
            "std_nnet_value": float(nnet_vals.std()),
            "frac_saturated_pos": float((nnet_vals > 0.95).mean()),  # near +1
            "frac_saturated_neg": float((nnet_vals < -0.95).mean()), # near -1
            "frac_saturated_any": float((np.abs(nnet_vals) > 0.95).mean()),
            # Prediction accuracy: does sign of prediction match target?
            "sign_accuracy": float((np.sign(nnet_vals) == np.sign(targets)).mean()),
            # Mean absolute error vs target
            "mae_vs_outcome": float(np.abs(nnet_vals - targets).mean()),
            # Per-player predictions
            "mean_when_x_moves": float(nnet_vals[players == -1].mean()) if (players == -1).any() else 0,
            "mean_when_o_moves": float(nnet_vals[players == 1].mean()) if (players == 1).any() else 0,
            # Correlation between prediction and target
            "pred_outcome_corr": float(np.corrcoef(nnet_vals, targets)[0, 1]) if len(nnet_vals) > 1 else 0,
            "n_predictions": len(all_preds),
        }

    def _run_simulations(self, roots, active):
        """Run num_simulations MCTS simulations for all active games.

        Uses K-select with virtual loss: each round selects K leaves per game,
        batches them for GPU eval, then backprops. Virtual loss ensures different
        selects explore different branches.
        """
        select_expand_time = 0.0
        backup_time = 0.0
        nn_time = 0.0
        terminal_hits = 0
        K = self.selects_per_round
        use_vl = K > 1

        for _ in range(0, self.num_simulations, K):
            pending = []  # (leaf, path) tuples

            t0 = time.time()
            if use_vl:
                for i in active:
                    for k in range(K):
                        leaf, path = self.mcts.search_expand_vl(roots[i], self.vl_value)
                        if leaf is not None:
                            pending.append((leaf, path))
                        else:
                            terminal_hits += 1
            else:
                # K=1, no VL needed — use original fast path
                for i in active:
                    leaf = self.mcts.search_expand(roots[i])
                    if leaf is not None:
                        pending.append((leaf, None))
                    else:
                        terminal_hits += 1
            select_expand_time += time.time() - t0

            if pending:
                if use_vl:
                    # Deduplicate by node id — same leaf selected twice
                    unique = {}
                    for leaf, path in pending:
                        nid = id(leaf)
                        if nid not in unique:
                            unique[nid] = (leaf, path)
                        else:
                            # Undo VL for duplicate
                            self.mcts.undo_virtual_loss(leaf, path, self.vl_value)
                            terminal_hits += 1  # count as handled

                    leaves = [lp[0] for lp in unique.values()]
                    paths = [lp[1] for lp in unique.values()]
                else:
                    leaves = [lp[0] for lp in pending]
                    paths = [None] * len(leaves)

                t0 = time.time()
                self._batch_evaluate_nodes(leaves)
                nn_time += time.time() - t0

                t0 = time.time()
                if use_vl:
                    for leaf, path in zip(leaves, paths):
                        self.mcts.search_backup_vl(leaf, path, self.vl_value)
                else:
                    for node in leaves:
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

        # Verify encoding consistency
        for node, inp in zip(nodes, state_inputs):
            self._encoding_checks += 1
            player = node.state.player
            num_hist = getattr(self.game, 'num_history_states', 2)
            c = 2 * num_hist  # channel offset for current board
            board = node.state.board

            errors = []
            # Piece count check: ch[c] = my pieces, ch[c+1] = opponent pieces
            my_pieces_board = (board == player).sum()
            opp_pieces_board = (board == -player).sum()
            ch_c = inp[c].sum()
            ch_c1 = inp[c + 1].sum()
            if ch_c != my_pieces_board:
                errors.append(f"my pieces: board={my_pieces_board} enc={ch_c}")
            if ch_c1 != opp_pieces_board:
                errors.append(f"opp pieces: board={opp_pieces_board} enc={ch_c1}")

            if errors:
                self._encoding_errors += 1
                if self._encoding_errors <= 5:
                    print(f"  [ENCODING ERROR] {'; '.join(errors)}")

        values, policies, detail = self.net.batch_predict(state_inputs, detailed_timing=True)

        t0 = time.time()
        for node, value, policy in zip(nodes, values, policies):
            node.resolve(value, policy)
        postprocess_time = time.time() - t0

        self._last_preprocess_time += preprocess_time
        self._last_transfer_time += detail["transfer_time"]
        self._last_forward_time += detail["forward_time"]
        self._last_result_time += detail["result_time"]
        self._last_postprocess_time += postprocess_time
        self._last_batch_count += 1
        self._last_sample_count += len(nodes)
        self._last_min_batch = min(self._last_min_batch, len(nodes))
        self._last_max_batch = max(self._last_max_batch, len(nodes))
