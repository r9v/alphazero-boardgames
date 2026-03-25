class TrainingLogger:
    """Console and TensorBoard logging for training health metrics."""

    def __init__(self, trainer):
        self._t = trainer
        self.writer = trainer.writer

    def close(self):
        self.writer.close()

    def log_iteration(self, iteration, num_iterations, stats):
        """Log core metrics for one iteration to console and TensorBoard."""
        writer = self.writer
        train_result = stats['train_result']
        d = self._t._train_diag if hasattr(self._t, '_train_diag') else {}

        if train_result is not None:
            avg_loss, avg_value_loss, avg_policy_loss = train_result
            writer.add_scalar("loss/total", avg_loss, iteration)
            writer.add_scalar("loss/value", avg_value_loss, iteration)
            writer.add_scalar("loss/policy", avg_policy_loss, iteration)
            print(f"  Iter {iteration+1}/{num_iterations}: loss={avg_loss:.4f} "
                  f"(v={avg_value_loss:.4f} p={avg_policy_loss:.4f}) | "
                  f"games: p1={stats['wins_p1']} p2={stats['wins_p2']} "
                  f"draw={stats['draws']} | "
                  f"avg_len={stats['avg_length']:.1f} "
                  f"({stats['min_length']}-{stats['max_length']}) | "
                  f"self_play={stats['self_play_time']:.1f}s "
                  f"train={stats['train_time']:.1f}s "
                  f"total={stats['iter_time']:.1f}s")

        writer.add_scalar("perf/self_play_time", stats['self_play_time'], iteration)
        writer.add_scalar("perf/train_time", stats['train_time'], iteration)

        if d:
            rb = d.get('rb_grad_norms', {})
            if rb:
                eff_lr = rb.get('0_eff_lr', 0)
                writer.add_scalar("diag/eff_lr_rb0", eff_lr, iteration)

            pv_std = d.get('pred_v_std', 0)
            writer.add_scalar("diag/pred_v_std", pv_std, iteration)

            sign_acc = 0
            if hasattr(self._t, '_batched') and hasattr(self._t._batched, 'value_diag'):
                vd = self._t._batched.value_diag
                if vd:
                    sign_acc = vd.get('sign_accuracy', 0)
            writer.add_scalar("diag/sign_acc", sign_acc, iteration)

            top1 = d.get('policy_top1_acc', 0)
            writer.add_scalar("diag/policy_top1_acc", top1, iteration)

            val_vloss = d.get('val_vloss', 0)
            train_vloss = d.get('avg_value_loss', 0)
            gap = val_vloss - train_vloss if val_vloss and train_vloss else 0
            writer.add_scalar("diag/vloss_gap", gap, iteration)

            print(f"  Diag: eff_lr_rb0={rb.get('0_eff_lr', 0):.4f} "
                  f"pred_v_std={pv_std:.3f} sign_acc={sign_acc:.1%} "
                  f"policy_top1={top1:.1%} vloss_gap={gap:+.4f} "
                  f"buf={d.get('buffer_fill', 0)}/{d.get('buffer_capacity', 0)}"
                  if rb else "")

            # MCTS correction metrics (from self-play)
            if hasattr(self._t, '_batched') and hasattr(self._t._batched, 'value_diag'):
                vd = self._t._batched.value_diag
                if vd:
                    mcts_corr = vd.get('mcts_nnet_corr', 0)
                    mcts_correction = vd.get('mcts_correction_mean', 0)
                    writer.add_scalar("diag/mcts_nnet_corr", mcts_corr, iteration)
                    writer.add_scalar("diag/mcts_correction", mcts_correction, iteration)
                    print(f"  Diag: mcts_corr={mcts_corr:.3f} mcts_correction={mcts_correction:+.3f}")

            # SVD rank (capacity indicator)
            svd = d.get('svd', {})
            if svd:
                writer.add_scalar("diag/svd_rank90", svd['rank90'], iteration)
                writer.add_scalar("diag/svd_rank99", svd['rank99'], iteration)
                print(f"  Diag: svd_rank90={svd['rank90']}/{svd['n_filters']} "
                      f"rank99={svd['rank99']}/{svd['n_filters']}")

            # Backbone drift
            drift = d.get('drift_cos')
            if drift is not None:
                writer.add_scalar("diag/backbone_drift_cos", drift, iteration)
                print(f"  Diag: backbone_drift_cos={drift:.4f}")

            # Game phase value loss
            gp = d.get('game_phase_vloss', {})
            if gp:
                parts = []
                for phase in ('early', 'mid', 'late'):
                    if phase in gp:
                        writer.add_scalar(f"diag/vloss_{phase}", gp[phase], iteration)
                        parts.append(f"{phase}={gp[phase]:.3f}")
                if parts:
                    print(f"  Diag: vloss_phase {' '.join(parts)}")
