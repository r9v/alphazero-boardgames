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
            print(f"  Not enough samples ({len(samples)}), skipping training")
            return None

        # === DIAGNOSTIC: Value target distribution ===
        all_values = np.array([s[2] for s in samples])
        val_mean = all_values.mean()
        val_std = all_values.std()
        frac_pos = (all_values > 0).mean()   # O wins
        frac_neg = (all_values < 0).mean()   # X wins
        frac_draw = (all_values == 0).mean()  # draws

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

        # Use fixed number of training steps with random sampling
        n_samples = len(samples)
        num_steps = min(self.max_train_steps, self.epochs * (n_samples // self.batch_size))
        effective_epochs = (num_steps * self.batch_size) / n_samples
        early_cutoff = max(num_steps // 10, 1)
        late_start = num_steps - early_cutoff

        for step in range(num_steps):
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

            self.buffer.insert_batch(all_examples)

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
                if enc_errs > 0:
                    print(f"  [WARNING] Encoding errors: {enc_errs}/{enc_total}")
                else:
                    print(f"  Encoding: {enc_total} checks, all OK")
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
                print(f"  Diag: eff_epochs={d['effective_epochs']:.1f} "
                      f"steps={d['num_steps']} | "
                      f"vloss early={d['early_vloss']:.4f} "
                      f"late={d['late_vloss']:.4f} "
                      f"(delta={d['late_vloss']-d['early_vloss']:+.4f}) | "
                      f"buf={d['buffer_fill']}/{d['buffer_capacity']}"
                      f"{' FULL' if d['buffer_full'] else ''}")
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
                    rel = getattr(self.game, 'relative_encoding', False)
                    for entry in glog:
                        player_str = "X" if entry["player"] == -1 else "O"
                        pi = entry["pi"]
                        best_action = max(range(len(pi)), key=lambda a: pi[a])
                        # Show NN value vs eventual outcome
                        nnet_v = entry["nnet_value"]
                        # For relative encoding, target is outcome from this player's perspective
                        target = outcome * entry["player"] if rel else outcome
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

        rel = getattr(self.game, 'relative_encoding', False)

        # Position 1: Empty board (X to move) - should be roughly neutral
        s = self.game.new_game()
        positions.append(("empty_board", self.game.state_to_input(s), "expect ~0"))

        # Position 2: X about to win horizontally (X to move)
        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = -1  # X has 3 in row
        board[0][4] = 1     # some O piece
        s = C4State(None, board, player=-1)
        # Relative: I'm X and winning -> expect > +0.5
        # Absolute: X winning -> expect < -0.5
        exp = "expect > +0.5 (I'm winning)" if rel else "expect < -0.5 (X wins)"
        positions.append(("x_wins_next", self.game.state_to_input(s), exp))

        # Position 3: O about to win horizontally (O to move)
        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = 1   # O has 3 in row
        board[0][4] = -1    # some X piece
        board[1][4] = -1
        s = C4State(None, board, player=1)
        # Relative: I'm O and winning -> expect > +0.5
        # Absolute: O winning -> expect > +0.5
        exp = "expect > +0.5 (I'm winning)" if rel else "expect > +0.5 (O wins)"
        positions.append(("o_wins_next", self.game.state_to_input(s), exp))

        # Position 4: The diagonal threat position from the bug report
        board = np.zeros((6, 7), dtype="int")
        board[0] = [0, 1, 1, -1, 0, 1, -1]
        board[1] = [0, -1, 1, -1, 0, 0, 0]
        board[2] = [0, 1, -1, 1, 0, 0, 0]
        board[3] = [0, -1, 0, -1, 0, 0, 0]
        s = C4State(None, board, player=1)  # O to move, X threatens diagonal
        # Relative: I'm O and losing -> expect < 0
        exp = "expect < 0 (I'm losing)" if rel else "expect < 0 (X winning)"
        positions.append(("diag_threat", self.game.state_to_input(s), exp))

        # Position 5: X has strong center control (X to move)
        board = np.zeros((6, 7), dtype="int")
        board[0][3] = -1
        board[1][3] = -1
        board[0][2] = 1
        board[0][4] = 1
        s = C4State(None, board, player=-1)
        # Relative: I'm X with slight advantage -> expect > 0
        exp = "expect > 0 (I'm slightly winning)" if rel else "expect < 0 (X slight advantage)"
        positions.append(("x_center", self.game.state_to_input(s), exp))

        return positions
