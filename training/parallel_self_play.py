import time
import random
import numpy as np

from mcts import MCTS, Node, add_dirichlet_noise
from utils import log_backends


def _finalize_game_targets(examples, tv, label=""):
    """Convert per-move player tags to outcome targets and verify sign chain.

    Args:
        examples: list of [state_input, policy, player_tag, aux_maps] for one game
        tv: terminal value (-1=X wins, +1=O wins, 0=draw)
        label: context string for assertion messages

    After finalization, each example is:
        [state_input, policy, value, aux_maps]
    where aux_maps is a dict with keys: policy_surprise, mcts_q
    """
    if tv != 0 and len(examples) > 0:
        assert examples[-1][2] == tv, \
            f"{label} sign-chain: last_player={examples[-1][2]}, tv={tv}"
    for ex in examples:
        player_at_pos = ex[2]
        ex[2] = tv * player_at_pos
    if tv != 0 and len(examples) > 0:
        assert abs(examples[-1][2] - 1.0) < 1e-6, \
            f"{label} target: last={examples[-1][2]:.4f}, expected +1"
        if len(examples) >= 2:
            assert abs(examples[-2][2] - (-1.0)) < 1e-6, \
                f"{label} target: 2nd_last={examples[-2][2]:.4f}, expected -1"


class BatchedSelfPlay:
    """Runs N self-play games in parallel with batched NN evaluation."""

    def __init__(self, game, net, num_games, num_simulations,
                 selects_per_round=1, vl_value=0.0,
                 temp_threshold=15, c_puct=1.5,
                 dirichlet_alpha=1.0, dirichlet_epsilon=0.25,
                 tree_reuse=True, resign_threshold=-1.0,
                 resign_min_moves=99, resign_check_prob=0.0,
                 random_opening_moves=0,
                 random_opening_fraction=1.0,
                 contempt_n=0):
        self.game = game
        self.net = net
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.selects_per_round = selects_per_round
        self.vl_value = vl_value
        self.temp_threshold = temp_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.tree_reuse = tree_reuse
        self.resign_threshold = resign_threshold
        self.resign_min_moves = resign_min_moves
        self.resign_check_prob = resign_check_prob
        self.random_opening_moves = random_opening_moves
        self.random_opening_fraction = random_opening_fraction
        self.mcts = MCTS(game, net, c_puct=c_puct, contempt_n=contempt_n)

        if not getattr(BatchedSelfPlay, '_backend_logged', False):
            log_backends(MCTS, game)
            BatchedSelfPlay._backend_logged = True

    def play_games(self):
        """Play num_games self-play games in parallel.

        Returns:
            (all_examples, results, game_lengths) where:
            - all_examples: flat list of [state_input, policy, outcome]
            - results: list of terminal_value per game
            - game_lengths: list of number of moves per game
        """
        self._p = {
            'select_expand_time': 0.0, 'backup_time': 0.0, 'nn_time': 0.0,
            'preprocess_time': 0.0, 'transfer_time': 0.0, 'forward_time': 0.0,
            'result_time': 0.0, 'postprocess_time': 0.0,
            'batch_count': 0, 'sample_count': 0, 'terminal_hits': 0,
            'min_batch': float('inf'), 'max_batch': 0,
            'encoding_checks': 0, 'encoding_errors': 0, 'encoding_time': 0.0,
            'batch_histogram': [0, 0, 0, 0, 0],  # [1-4, 5-16, 17-32, 33-64, 65+]
            'active_per_move': [],
            'accum_rounds': 0,
            'tree_reuse_count': 0, 'tree_reuse_fresh_count': 0,
            'tree_reuse_visits_sum': 0,
            'resign_count': 0, 'resign_move_sum': 0,
            'resign_false_positives': 0, 'resign_check_count': 0,
            'imm_win_count': 0, 'imm_win_total': 0,
            'random_opening_total': 0, 'random_opening_terminated': 0,
            'random_opening_surviving': self.num_games,
            'col_selections': [0] * self.game.action_size,  # per-column move count
            'win_patterns': {},  # {(type, col): count} e.g. ('vert', 0): 5
        }

        self._game_value_preds = [[] for _ in range(self.num_games)]  # (player, nnet_value, mcts_Q) per move
        self._mcts_visit_entropies = []  # entropy of MCTS visit distribution per move

        states = [self.game.new_game() for _ in range(self.num_games)]
        examples = [[] for _ in range(self.num_games)]
        raw_priors = {}
        terminal_values = [0] * self.num_games
        active = list(range(self.num_games))
        move_counts = [0] * self.num_games

        random_opening_counts = [0] * self.num_games
        if self.random_opening_moves > 0:
            total_random = 0
            total_terminated = 0
            frac = getattr(self, 'random_opening_fraction', 1.0)
            for i in range(self.num_games):
                if random.random() >= frac:
                    continue
                k = random.randint(0, self.random_opening_moves)
                for m in range(k):
                    legal = states[i].available_actions
                    legal_indices = [a for a in range(len(legal)) if legal[a]]
                    if not legal_indices or states[i].terminal:
                        break
                    random.shuffle(legal_indices)
                    action = legal_indices[0]
                    for a in legal_indices:
                        ns = self.game.step(states[i], a)
                        if not ns.terminal:
                            action = a
                            break
                    states[i] = self.game.step(states[i], action)
                    random_opening_counts[i] += 1
                    if states[i].terminal:
                        total_terminated += 1
                        break
                total_random += random_opening_counts[i]
            surviving = [i for i in range(self.num_games)
                         if not states[i].terminal]
            self._p['random_opening_total'] = total_random
            self._p['random_opening_terminated'] = total_terminated
            self._p['random_opening_surviving'] = len(surviving)
            for i in range(self.num_games):
                if states[i].terminal:
                    terminal_values[i] = states[i].terminal_value
            active = surviving
        else:
            self._p['random_opening_total'] = 0
            self._p['random_opening_terminated'] = 0
            self._p['random_opening_surviving'] = self.num_games

        resign_allowed = [random.random() >= self.resign_check_prob
                          for _ in range(self.num_games)]
        resign_check_player = [0] * self.num_games

        roots = [Node(None, s, self.game) if i in active else None
                 for i, s in enumerate(states)]
        active_roots = [roots[i] for i in active]
        self._batch_evaluate_nodes(active_roots)

        for i in active:
            raw_priors[i] = roots[i].P.copy() if roots[i].P is not None else None
        for root in active_roots:
            root.P = add_dirichlet_noise(root.P, self.dirichlet_alpha, self.dirichlet_epsilon, root.available_actions_mask)

        while active:
            self._p['active_per_move'].append(len(active))
            self._run_simulations(roots, active)

            still_active = []
            resign_enabled = self.resign_threshold > -1.0
            for i in active:
                root_value = -roots[i].Q  # positive = winning, negative = losing
                should_resign = (
                    resign_enabled
                    and move_counts[i] >= self.resign_min_moves
                    and root_value < self.resign_threshold
                )
                if should_resign and resign_allowed[i]:
                    tv = -states[i].player
                    terminal_values[i] = tv
                    _finalize_game_targets(examples[i], tv, label="Resign")
                    self._track_win_pattern(states[i].board, tv)
                    self._p['resign_count'] += 1
                    self._p['resign_move_sum'] += move_counts[i]
                elif should_resign and not resign_allowed[i]:
                    if resign_check_player[i] == 0:
                        resign_check_player[i] = states[i].player
                        self._p['resign_check_count'] += 1
                    still_active.append(i)
                else:
                    still_active.append(i)

            next_active_fresh = []
            next_active_reused = []
            for i in still_active:
                root = roots[i]
                pi = np.zeros(np.shape(root.available_actions_mask))
                children = root.children
                for action in root.available_actions:
                    child = children[action]
                    if child is not None:
                        pi[action] = child.n / root.n
                pi_sum = pi.sum()
                if pi_sum > 0:
                    pi = pi / pi_sum

                # Policy target: subtract noise-induced visits from non-best children
                best_action_raw = np.argmax(pi)
                pi_target = np.zeros_like(pi)
                for action in root.available_actions:
                    child = children[action]
                    if child is None:
                        continue
                    raw_visits = child.n
                    if action != best_action_raw:
                        rp = raw_priors.get(i)
                        prior_val = rp[action] if rp is not None else 0.0
                        noise_visits = (2.0 * prior_val * root.n) ** 0.5
                        raw_visits = max(0, raw_visits - noise_visits)
                    pi_target[action] = raw_visits
                pt_sum = pi_target.sum()
                if pt_sum > 0:
                    pi_target = pi_target / pt_sum
                else:
                    pi_target = pi

                pi_nz = pi[pi > 0]
                if len(pi_nz) > 0:
                    self._mcts_visit_entropies.append(
                        float(-np.sum(pi_nz * np.log(pi_nz)))
                    )

                self._game_value_preds[i].append((states[i].player, root.nnet_value, -root.Q))

                move_num = len(examples[i])
                if move_num < self.temp_threshold:
                    action = np.random.choice(len(pi), p=pi)
                else:
                    action = np.argmax(pi)
                self._p['imm_win_total'] += 1
                for a in range(len(states[i].available_actions)):
                    if states[i].available_actions[a]:
                        ns = self.game.step(states[i], a)
                        if ns.terminal and ns.terminal_value != 0:
                            self._p['imm_win_count'] += 1
                            break

                self._p['col_selections'][action] += 1

                prior = root.P
                kl_div = 0.0
                if prior is not None:
                    eps = 1e-8
                    pi_safe = np.clip(pi, eps, 1.0)
                    prior_safe = np.clip(prior, eps, 1.0)
                    mask = pi > eps
                    if mask.any():
                        kl_div = float(np.sum(pi_safe[mask] * np.log(pi_safe[mask] / prior_safe[mask])))

                mcts_q = float(-root.Q) if root.n > 0 else 0.0

                aux_maps = {
                    'policy_surprise': kl_div,
                    'mcts_q': mcts_q,
                }
                examples[i].append([self.game.state_to_input(states[i]), pi_target, states[i].player, aux_maps])
                move_counts[i] += 1

                states[i] = self.game.step(states[i], action)

                if states[i].terminal:
                    tv = states[i].terminal_value
                    terminal_values[i] = tv
                    if resign_check_player[i] != 0:
                        resigner_outcome = tv * resign_check_player[i]
                        if resigner_outcome >= 0:
                            self._p['resign_false_positives'] += 1
                    _finalize_game_targets(examples[i], tv, label="Terminal")
                    self._track_win_pattern(states[i].board, tv)
                else:
                    if self.tree_reuse:
                        subtree = root.children[action]
                        if subtree is not None and subtree.P is not None:
                            self._p['tree_reuse_visits_sum'] += subtree.n
                            subtree.parent = None
                            roots[i] = subtree
                            next_active_reused.append(i)
                            self._p['tree_reuse_count'] += 1
                        else:
                            roots[i] = Node(None, states[i], self.game)
                            next_active_fresh.append(i)
                            self._p['tree_reuse_fresh_count'] += 1
                    else:
                        roots[i] = Node(None, states[i], self.game)
                        next_active_fresh.append(i)

            if next_active_fresh:
                new_roots = [roots[i] for i in next_active_fresh]
                self._batch_evaluate_nodes(new_roots)
                for i in next_active_fresh:
                    raw_priors[i] = roots[i].P.copy() if roots[i].P is not None else None
                    roots[i].P = add_dirichlet_noise(roots[i].P, self.dirichlet_alpha, self.dirichlet_epsilon, roots[i].available_actions_mask)

            for i in next_active_reused:
                raw_priors[i] = roots[i].P.copy() if roots[i].P is not None else None
                roots[i].P = add_dirichlet_noise(roots[i].P, self.dirichlet_alpha, self.dirichlet_epsilon, roots[i].available_actions_mask)

            active = next_active_fresh + next_active_reused

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

        # Build perf dict: copy counters + compute derived values
        p = self._p
        self.perf = dict(p)
        self.perf['min_batch'] = p['min_batch'] if p['min_batch'] != float('inf') else 0
        self.perf['tree_reuse_avg_visits'] = p['tree_reuse_visits_sum'] / max(p['tree_reuse_count'], 1)
        self.perf['resign_avg_move'] = p['resign_move_sum'] / max(p['resign_count'], 1)
        self.perf['imm_win_frac'] = p['imm_win_count'] / max(p['imm_win_total'], 1)

        # Compute self-play value prediction diagnostics
        self._compute_value_diagnostics(results)

        return all_examples, results, game_lengths

    def _track_win_pattern(self, board, tv):
        """Classify and count win patterns if the game has a winner."""
        if tv == 0:
            return  # draw
        try:
            from games.connect4 import classify_win
            wins = classify_win(board)
            for win_type, col, _player in wins:
                key = (win_type, col)
                self._p['win_patterns'][key] = self._p['win_patterns'].get(key, 0) + 1
        except (ImportError, AttributeError):
            pass  # not Connect4, skip

    def _compute_value_diagnostics(self, results):
        """Compute statistics about NN value predictions during self-play."""
        all_preds = []   # (nnet_value, target_outcome, player, mcts_q)
        for i, game_preds in enumerate(self._game_value_preds):
            outcome = results[i]
            for player, nnet_v, mcts_q in game_preds:
                # Relative: compare nnet against target from player's perspective
                target = outcome * player
                all_preds.append((nnet_v, target, player, mcts_q))

        if not all_preds:
            self.value_diag = {}
            return

        nnet_vals = np.array([p[0] for p in all_preds])
        targets = np.array([p[1] for p in all_preds])
        players = np.array([p[2] for p in all_preds])
        mcts_q_vals = np.array([p[3] for p in all_preds])

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
            # MCTS visit entropy
            "mcts_visit_entropy_mean": float(np.mean(self._mcts_visit_entropies)) if self._mcts_visit_entropies else 0.0,
            "mcts_visit_entropy_std": float(np.std(self._mcts_visit_entropies)) if self._mcts_visit_entropies else 0.0,
            # MCTS Q vs nnet value agreement
            "mcts_nnet_corr": float(np.corrcoef(nnet_vals, mcts_q_vals)[0, 1]) if len(nnet_vals) > 1 else 0.0,
            "mcts_nnet_mae": float(np.abs(nnet_vals - mcts_q_vals).mean()),
            "mcts_correction_mean": float((mcts_q_vals - nnet_vals).mean()),
            "mcts_q_mean": float(mcts_q_vals.mean()),
            "mcts_q_std": float(mcts_q_vals.std()),
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
            self._p['accum_rounds'] += 1

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

        self._p['select_expand_time'] += select_expand_time
        self._p['backup_time'] += backup_time
        self._p['nn_time'] += nn_time
        self._p['terminal_hits'] += terminal_hits

    def _batch_evaluate_nodes(self, nodes):
        """Batch neural network evaluation for a list of nodes."""
        if not nodes:
            return

        t0 = time.time()
        state_inputs = [self.game.state_to_input(node.state) for node in nodes]
        preprocess_time = time.time() - t0

        # Verify encoding consistency (sample 1 in 50 to reduce overhead)
        # Checks that channels 0/1 match my/opponent pieces (standard
        # relative encoding used by Connect4 and TicTacToe).
        # Uses arithmetic to find the check index within this batch,
        # avoiding a Python loop over all nodes.
        t_enc = time.time()
        prev = self._p['encoding_checks']
        batch_len = len(nodes)
        self._p['encoding_checks'] += batch_len
        # Find the first check index: next multiple of 50 after prev
        next_check = ((prev // 50) + 1) * 50
        while next_check < prev + batch_len:
            try:
                idx = next_check - prev  # index within this batch
                node = nodes[idx]
                inp = state_inputs[idx]
                player = node.state.player
                num_hist = getattr(self.game, 'num_history_states', 0)
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
                    self._p['encoding_errors'] += 1
                    if self._p['encoding_errors'] <= 5:
                        print(f"  [ENCODING ERROR] {'; '.join(errors)}")
            except (IndexError, ValueError, TypeError):
                pass  # encoding layout doesn't match standard pattern
            next_check += 50
        self._p['encoding_time'] += time.time() - t_enc

        values, policies, detail = self.net.batch_predict(state_inputs, detailed_timing=True)

        t0 = time.time()
        for node, value, policy in zip(nodes, values, policies):
            node.resolve(value, policy)
        postprocess_time = time.time() - t0

        self._p['preprocess_time'] += preprocess_time
        self._p['transfer_time'] += detail["transfer_time"]
        self._p['forward_time'] += detail["forward_time"]
        self._p['result_time'] += detail["result_time"]
        self._p['postprocess_time'] += postprocess_time
        self._p['batch_count'] += 1
        self._p['sample_count'] += len(nodes)
        self._p['min_batch'] = min(self._p['min_batch'], len(nodes))
        self._p['max_batch'] = max(self._p['max_batch'], len(nodes))
        # Batch histogram: [1-4, 5-16, 17-32, 33-64, 65+]
        bs = len(nodes)
        if bs <= 4:
            self._p['batch_histogram'][0] += 1
        elif bs <= 16:
            self._p['batch_histogram'][1] += 1
        elif bs <= 32:
            self._p['batch_histogram'][2] += 1
        elif bs <= 64:
            self._p['batch_histogram'][3] += 1
        else:
            self._p['batch_histogram'][4] += 1
