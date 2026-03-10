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
                 selects_per_round=1, vl_value=0.0,
                 temp_threshold=15, c_puct=1.5):
        self.game = game
        self.net = net
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.selects_per_round = selects_per_round
        self.vl_value = vl_value
        self.temp_threshold = temp_threshold
        self.mcts = MCTS(game, net, c_puct=c_puct)

        # Log backends once
        if not getattr(BatchedSelfPlay, '_backend_logged', False):
            mcts_mod = MCTS.__module__
            mcts_label = "C/Cython" if "c_mcts" in mcts_mod else "Python"
            game_mod = type(game).__module__
            game_label = "C/Cython" if "c_game" in game_mod else "Python"
            print(f"  MCTS backend: {mcts_label} ({mcts_mod})")
            print(f"  Game backend: {game_label} ({game_mod})")
            BatchedSelfPlay._backend_logged = True

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
        self._last_encoding_time = 0.0
        self._batch_histogram = [0, 0, 0, 0, 0]  # [1-4, 5-16, 17-32, 33-64, 65+]
        self._active_per_move = []  # track len(active) each move step
        self._accum_rounds = 0  # how many times accumulation path fired

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
            self._active_per_move.append(len(active))
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
                    total_moves = len(examples[i])
                    for move_idx, ex in enumerate(examples[i]):
                        player_at_pos = ex[2]
                        # Relative: target from current player's perspective
                        raw_target = tv * player_at_pos
                        # Discount by distance from end: early moves get smaller targets
                        # γ^(moves_to_end) where γ=0.95
                        moves_to_end = total_moves - 1 - move_idx
                        discount = 0.95 ** moves_to_end
                        target = raw_target * discount
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
            "encoding_time": self._last_encoding_time,
            "batch_histogram": self._batch_histogram,
            "active_per_move": self._active_per_move,
            "accum_rounds": self._accum_rounds,
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
            # Exclude draws (target=0) since np.sign(0)=0 always mismatches
            "sign_accuracy": float(
                (np.sign(nnet_vals[targets != 0]) == np.sign(targets[targets != 0])).mean()
            ) if (targets != 0).any() else 0.5,
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

        When few games remain active (< 8), accumulates leaves across multiple
        simulation rounds before flushing to GPU to avoid tiny batches.
        """
        select_expand_time = 0.0
        backup_time = 0.0
        nn_time = 0.0
        terminal_hits = 0
        K = self.selects_per_round
        use_vl = K > 1
        n_active = len(active)
        MIN_BATCH_TARGET = 8

        # Accumulation mode: when few games remain, gather multiple rounds
        # to build a decent batch before hitting GPU
        use_accum = n_active < MIN_BATCH_TARGET
        if use_accum:
            accum_vl = 3.0  # temporary VL for diverse selection during accumulation
            rounds_per_flush = max(1, (MIN_BATCH_TARGET + n_active - 1) // n_active)
            self._accum_rounds += 1

        sims_done = 0
        while sims_done < self.num_simulations:
            if use_accum:
                # Accumulation path: gather multiple rounds of leaves with VL
                all_pending = []  # (leaf, path) tuples across rounds
                rounds_this_flush = min(rounds_per_flush,
                                        (self.num_simulations - sims_done + K - 1) // K)
                t0 = time.time()
                for _r in range(rounds_this_flush):
                    for i in active:
                        for _k in range(K):
                            leaf, path = self.mcts.search_expand_vl(roots[i], accum_vl)
                            if leaf is not None:
                                all_pending.append((leaf, path))
                            else:
                                terminal_hits += 1
                    sims_done += K
                select_expand_time += time.time() - t0

                if all_pending:
                    # Deduplicate
                    unique = {}
                    for leaf, path in all_pending:
                        nid = id(leaf)
                        if nid not in unique:
                            unique[nid] = (leaf, path)
                        else:
                            self.mcts.undo_virtual_loss(leaf, path, accum_vl)
                            terminal_hits += 1

                    leaves = [lp[0] for lp in unique.values()]
                    paths = [lp[1] for lp in unique.values()]

                    t0 = time.time()
                    self._batch_evaluate_nodes(leaves)
                    nn_time += time.time() - t0

                    t0 = time.time()
                    for leaf, path in zip(leaves, paths):
                        self.mcts.search_backup_vl(leaf, path, accum_vl)
                    backup_time += time.time() - t0
            else:
                # Normal path: one round per flush
                pending = []

                t0 = time.time()
                if use_vl:
                    for i in active:
                        for _k in range(K):
                            leaf, path = self.mcts.search_expand_vl(roots[i], self.vl_value)
                            if leaf is not None:
                                pending.append((leaf, path))
                            else:
                                terminal_hits += 1
                else:
                    for i in active:
                        leaf = self.mcts.search_expand(roots[i])
                        if leaf is not None:
                            pending.append((leaf, None))
                        else:
                            terminal_hits += 1
                select_expand_time += time.time() - t0
                sims_done += K

                if pending:
                    if use_vl:
                        unique = {}
                        for leaf, path in pending:
                            nid = id(leaf)
                            if nid not in unique:
                                unique[nid] = (leaf, path)
                            else:
                                self.mcts.undo_virtual_loss(leaf, path, self.vl_value)
                                terminal_hits += 1
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

        # Verify encoding consistency (sample 1 in 50 to reduce overhead)
        # Only applies to games using the default history-based encoding
        # (2 channels per history step). Games with custom input_channels
        # (e.g. Santorini) use a different layout and skip this check.
        t_enc = time.time()
        if not hasattr(self.game, 'input_channels'):
            for node, inp in zip(nodes, state_inputs):
                self._encoding_checks += 1
                if self._encoding_checks % 50 != 0:
                    continue
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
        self._last_encoding_time += time.time() - t_enc

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
        # Batch histogram: [1-4, 5-16, 17-32, 33-64, 65+]
        bs = len(nodes)
        if bs <= 4:
            self._batch_histogram[0] += 1
        elif bs <= 16:
            self._batch_histogram[1] += 1
        elif bs <= 32:
            self._batch_histogram[2] += 1
        elif bs <= 64:
            self._batch_histogram[3] += 1
        else:
            self._batch_histogram[4] += 1
