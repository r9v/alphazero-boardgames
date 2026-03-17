import numpy as np
import torch
import torch.nn.functional as F


class TrainingLogger:
    """Handles all console and TensorBoard logging for the training loop.

    Extracted from Trainer to separate logging concerns from training logic.
    Accesses Trainer state via self._t reference.
    """

    # Data-driven TensorBoard scalar mappings for value head diagnostics.
    # Each entry is (tb_tag, vh_key). Missing keys are silently skipped.
    _VH_SCALARS = [
        ("vh/dead_neurons", "dead_neurons"), ("vh/active_neurons", "active_neurons"),
        ("vh/wdl_entropy", "wdl_entropy"), ("vh/wdl_confidence", "wdl_confidence"),
        ("vh/wdl_accuracy", "wdl_accuracy"), ("vh/wdl_logit_std", "wdl_logit_std"),
        ("vh/wdl_logit_range", "wdl_logit_range"),
        ("vh/wdl_win_prob", "wdl_win_prob"), ("vh/wdl_draw_prob", "wdl_draw_prob"),
        ("vh/wdl_loss_prob", "wdl_loss_prob"),
        ("vh/wdl_scalar_mean", "wdl_scalar_mean"), ("vh/wdl_scalar_std", "wdl_scalar_std"),
        ("vh/fc2_w_max", "fc2_w_max"), ("vh/fc2_w_min", "fc2_w_min"),
        ("vh/fc2_w_norm", "fc2_w_norm"), ("vh/fc1_w_norm", "fc1_w_norm"),
        ("vh/backbone_std", "backbone_std"),
        ("vh/fc1_act_p10", "fc1_act_p10"), ("vh/fc1_act_p50", "fc1_act_p50"),
        ("vh/fc1_act_p90", "fc1_act_p90"),
        ("vh/vconv_dead_channels", "vconv_dead_channels"),
        ("vh/fc2_w_norm_delta", "fc2_w_norm_delta"), ("vh/fc1_w_norm_delta", "fc1_w_norm_delta"),
        ("vh/grad_dead_mean", "grad_dead_mean"), ("vh/grad_alive_mean", "grad_alive_mean"),
        ("vc/backbone_raw_abs", "backbone_raw_abs"), ("vc/vconv_pre_bn_abs", "vconv_pre_bn_abs"),
        ("vc/vbn_post_abs", "vbn_post_abs"), ("vc/bn_ratio", "bn_ratio"),
        ("vc/vc_w_norm", "vc_w_norm"), ("vc/vbn_gamma_min", "vbn_gamma_min"),
        ("vc/vbn_gamma_mean", "vbn_gamma_mean"), ("vc/pconv_abs", "pconv_abs"),
        ("vc/vconv_grad_norm", "vconv_grad_norm"),
        ("bb/dead_channels", "bb_dead_channels"),
        ("bb/ch_p10", "bb_ch_p10"), ("bb/ch_p50", "bb_ch_p50"),
        ("bb/ch_p90", "bb_ch_p90"),
        ("bb/v_grad_norm", "bb_v_grad_norm"), ("bb/p_grad_norm", "bb_p_grad_norm"),
        ("bb/grad_ratio", "bb_grad_ratio"),
        ("bb/n_value_dom", "bb_n_value_dom"), ("bb/n_policy_dom", "bb_n_policy_dom"),
        ("bb/param_v_grad", "bb_param_v_grad"), ("bb/param_p_grad", "bb_param_p_grad"),
        ("bb/param_grad_ratio", "bb_param_grad_ratio"),
        ("diag/bb_grad_cosine_sim", "bb_grad_cosine_sim"),
        ("diag/bb_grad_conflict_channels", "bb_grad_conflict_channels"),
        ("vp/weight_corr", "vp_weight_corr"), ("vp/overlap_20", "vp_overlap_20"),
        ("vp/val_top20_act", "vp_val_top20_act"), ("vp/pol_top20_act", "vp_pol_top20_act"),
        ("vp/val_health_corr", "vp_val_health_corr"), ("vp/pol_health_corr", "vp_pol_health_corr"),
        ("svd/bb_rank90", "svd_bb_rank90"), ("svd/bb_rank99", "svd_bb_rank99"),
        ("svd/bb_near_zero", "svd_bb_near_zero"), ("svd/gn_dead_deepest", "gn_dead_deepest"),
        ("svd/pfc_rank90", "svd_pfc_rank90"), ("svd/pfc_rank99", "svd_pfc_rank99"),
        ("diag/final_bn_dead", "final_bn_dead"),
        ("diag/final_bn_gamma_mean", "final_bn_gamma_mean"),
        ("diag/final_bn_gamma_std", "final_bn_gamma_std"),
        ("diag/init_conv_w_norm", "init_conv_w_norm"),
    ]

    _VH_PER_BLOCK = [
        ("all_rb_bn", [
            ("bn2_dead", "dead"), ("bn2_neg_gamma", "neg_gamma"),
            ("svd_rank90", "svd_rank90"), ("bn2_gamma_mean", "gamma_mean"),
        ]),
        ("rb_act_stats", [
            ("act_abs_mean", "abs_mean"), ("act_std", "std"),
            ("act_dead_channels", "dead_channels"),
        ]),
        ("rb_conv_norms", [
            ("conv1_w_norm", "conv1"), ("conv2_w_norm", "conv2"),
            ("conv2_w_delta", "c2_delta"),
        ]),
        ("rb_bn2_stats", [
            ("bn2_out_abs", "bn2_out_abs"), ("bn2_out_std", "bn2_out_std"),
            ("conv2_raw_var", "conv2_raw_var"), ("conv2_raw_abs", "conv2_raw_abs"),
            ("bn2_batch_var", "bn2_batch_var_mean"),
        ]),
        ("rb_res_rank", [("res_rank90", "rank90"), ("res_rank99", "rank99")]),
    ]

    def __init__(self, trainer):
        self._t = trainer
        self.writer = trainer.writer

    def close(self):
        self.writer.close()

    def log_iteration(self, iteration, num_iterations, stats):
        """Log all metrics for one iteration to console and TensorBoard."""
        writer = self.writer
        train_result = stats['train_result']

        self._log_self_play_stats(iteration, stats)

        # Main iteration summary
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

        self._log_mcts_perf(iteration)
        self._log_training_perf(iteration)
        self._log_training_diagnostics(iteration, stats)
        self._log_selfplay_value_diagnostics(iteration)
        self._log_intra_iteration_dynamics(iteration)
        self.eval_diagnostic_positions(iteration)

    def _log_self_play_stats(self, iteration, stats):
        """Log 3-in-a-row, self-play counts, pre-seg, drift, weight delta."""
        writer = self.writer

        iter_3r = stats.get('iter_3r')
        if iter_3r:
            _mine = iter_3r['mine']
            _opp = iter_3r['opp']
            print(f"  Diag[3R-iter]: fresh batch ch0: n={_mine['count']} target={_mine['mean']:+.3f} | "
                  f"ch1: n={_opp['count']} target={_opp['mean']:+.3f}")
            writer.add_scalar("diag/iter_3r_ch0_target", _mine['mean'], iteration)
            writer.add_scalar("diag/iter_3r_ch1_target", _opp['mean'], iteration)
            iter_3r_sign = stats.get('iter_3r_sign')
            if iter_3r_sign:
                print(f"  Diag[SIGN-iter]: ch0_3row: +:{iter_3r_sign['pos']} -:{iter_3r_sign['neg']} "
                      f"0:{iter_3r_sign['zero']} (n={iter_3r_sign['n']})")

        writer.add_scalar("self_play/avg_game_length", stats['avg_length'], iteration)
        writer.add_scalar("self_play/wins_p1", stats['wins_p1'], iteration)
        writer.add_scalar("self_play/wins_p2", stats['wins_p2'], iteration)
        writer.add_scalar("self_play/draws", stats['draws'], iteration)
        writer.add_scalar("self_play/p1_win_pct", stats['p1_win_pct'], iteration)
        writer.add_scalar("self_play/buffer_size", len(self._t.buffer), iteration)

        pre_seg = stats.get('pre_seg')
        if pre_seg:
            print(f"  PreSeg: vp_corr={pre_seg['vp_corr']:.3f} overlap={pre_seg['overlap']}/20")
            writer.add_scalar("pre_seg/vp_corr", pre_seg['vp_corr'], iteration)
            writer.add_scalar("pre_seg/overlap_20", pre_seg['overlap'], iteration)

        drift = stats.get('drift')
        if drift:
            _cos_detail = " ".join(
                f"{nm}={drift['per_pos'][nm]:.4f}" for nm in drift['pos_names'])
            print(f"  Diag[DRIFT]: backbone cosine_sim mean={drift['cos_mean']:.4f} "
                  f"min={drift['cos_min']:.4f} | {_cos_detail}")
            writer.add_scalar("drift/bb_cosine_mean", drift['cos_mean'], iteration)
            writer.add_scalar("drift/bb_cosine_min", drift['cos_min'], iteration)
            for nm, val in drift['per_pos'].items():
                writer.add_scalar(f"drift/{nm}_cosine", val, iteration)

        wdelta = stats.get('wdelta')
        if wdelta:
            _wd_parts = [f"rb{bi}={rd:.4f}" for bi, rd in sorted(wdelta.items())]
            print(f"  Diag[WDELTA]: {' '.join(_wd_parts)}")
            for bi, rd in wdelta.items():
                writer.add_scalar(f"wdelta/rb{bi}_rel", rd, iteration)

    def _log_mcts_perf(self, iteration):
        """Log MCTS and NN inference performance metrics."""
        t = self._t
        if not (hasattr(t, '_batched') and hasattr(t._batched, 'perf')):
            return
        writer = self.writer
        perf = t._batched.perf
        avg_batch = perf["sample_count"] / max(perf["batch_count"], 1)
        writer.add_scalar("perf/mcts_select_expand", perf["select_expand_time"], iteration)
        writer.add_scalar("perf/mcts_backup", perf["backup_time"], iteration)
        writer.add_scalar("perf/nn_time", perf["nn_time"], iteration)
        writer.add_scalar("perf/nn_preprocess", perf["preprocess_time"], iteration)
        writer.add_scalar("perf/nn_transfer", perf["transfer_time"], iteration)
        writer.add_scalar("perf/nn_forward", perf["forward_time"], iteration)
        writer.add_scalar("perf/nn_result", perf["result_time"], iteration)
        writer.add_scalar("perf/nn_postprocess", perf["postprocess_time"], iteration)
        writer.add_scalar("perf/batch_count", perf["batch_count"], iteration)
        writer.add_scalar("perf/avg_batch_size", avg_batch, iteration)
        writer.add_scalar("perf/terminal_hits", perf["terminal_hits"], iteration)
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
        enc_time = perf.get("encoding_time", 0)
        enc_sampled = enc_total // 50 if enc_total else 0
        if enc_errs > 0:
            print(f"  [WARNING] Encoding errors: {enc_errs}/{enc_sampled} sampled "
                  f"({enc_total} total) {enc_time:.2f}s")
        else:
            print(f"  Encoding: {enc_sampled}/{enc_total} sampled, all OK, {enc_time:.2f}s")
        hist = perf.get("batch_histogram", [0]*5)
        print(f"  BatchHist: [1-4]={hist[0]} [5-16]={hist[1]} "
              f"[17-32]={hist[2]} [33-64]={hist[3]} [65+]={hist[4]}")
        apm = perf.get("active_per_move", [])
        if apm:
            print(f"  ActiveGames: min={min(apm)} avg={np.mean(apm):.1f} "
                  f"max={max(apm)} steps={len(apm)}")
        accum = perf.get("accum_rounds", 0)
        if accum > 0:
            print(f"  Accumulation: {accum} move-steps used batch accumulation")
        tr_count = perf.get("tree_reuse_count", 0)
        tr_fresh = perf.get("tree_reuse_fresh_count", 0)
        tr_total = tr_count + tr_fresh
        if tr_total > 0:
            tr_pct = tr_count / tr_total
            tr_avg_v = perf.get("tree_reuse_avg_visits", 0)
            print(f"  TreeReuse: {tr_count}/{tr_total} ({tr_pct:.0%}) "
                  f"avg_reused_visits={tr_avg_v:.1f}")
            writer.add_scalar("perf/tree_reuse_pct", tr_pct, iteration)
            writer.add_scalar("perf/tree_reuse_avg_visits", tr_avg_v, iteration)
        resign_count = perf.get("resign_count", 0)
        resign_checks = perf.get("resign_check_count", 0)
        resign_fp = perf.get("resign_false_positives", 0)
        if resign_count > 0 or resign_checks > 0:
            resign_avg_move = perf.get("resign_avg_move", 0)
            total_games = t.games_per_iteration
            resign_pct = resign_count / max(total_games, 1)
            fp_str = f"false_pos={resign_fp}/{resign_checks}" if resign_checks > 0 else "no_checks"
            print(f"  Resign: {resign_count}/{total_games} ({resign_pct:.0%}) "
                  f"avg_move={resign_avg_move:.1f} "
                  f"{fp_str}")
            writer.add_scalar("self_play/resign_count", resign_count, iteration)
            writer.add_scalar("self_play/resign_pct", resign_pct, iteration)
            writer.add_scalar("self_play/resign_avg_move", resign_avg_move, iteration)
            if resign_checks > 0:
                fp_rate = resign_fp / resign_checks
                writer.add_scalar("self_play/resign_fp_rate", fp_rate, iteration)
        imm_win_n = perf.get("imm_win_count", 0)
        imm_win_frac = perf.get("imm_win_frac", 0)
        imm_win_total = imm_win_n / max(imm_win_frac, 1e-9) if imm_win_frac > 0 else 0
        print(f"  ImmWin: {imm_win_n}/{int(imm_win_total)} ({imm_win_frac:.1%}) positions have immediate winning move")
        writer.add_scalar("self_play/imm_win_frac", imm_win_frac, iteration)
        writer.add_scalar("self_play/imm_win_count", imm_win_n, iteration)

    def _log_training_perf(self, iteration):
        """Log training data prep and gradient computation times."""
        t = self._t
        if not hasattr(t, '_train_perf'):
            return
        writer = self.writer
        tp = t._train_perf
        writer.add_scalar("perf/train_data_prep", tp["data_prep_time"], iteration)
        writer.add_scalar("perf/train_gradient", tp["gradient_time"], iteration)
        writer.add_scalar("perf/train_num_samples", tp["num_samples"], iteration)
        writer.add_scalar("perf/train_num_batches", tp["num_batches"], iteration)
        print(f"  Train: data={tp['data_prep_time']:.1f}s "
              f"grad={tp['gradient_time']:.1f}s | "
              f"samples={tp['num_samples']} "
              f"batches={tp['num_batches']}")

    def _log_training_diagnostics(self, iteration, stats):
        """Log all training diagnostic metrics to console and TensorBoard."""
        t = self._t
        if not hasattr(t, '_train_diag'):
            return
        writer = self.writer
        d = t._train_diag
        p1_win_pct = stats['p1_win_pct']
        avg_length = stats['avg_length']
        min_length, max_length = stats['min_length'], stats['max_length']

        # TensorBoard scalars
        writer.add_scalar("diag/val_target_mean", d["val_target_mean"], iteration)
        writer.add_scalar("diag/val_target_std", d["val_target_std"], iteration)
        writer.add_scalar("diag/frac_xwins", d["frac_neg"], iteration)
        writer.add_scalar("diag/frac_owins", d["frac_pos"], iteration)
        writer.add_scalar("diag/frac_draws", d["frac_draw"], iteration)
        writer.add_scalar("diag/effective_epochs", d["effective_epochs"], iteration)
        writer.add_scalar("diag/early_vloss", d["early_vloss"], iteration)
        writer.add_scalar("diag/late_vloss", d["late_vloss"], iteration)
        writer.add_scalar("diag/early_ploss", d["early_ploss"], iteration)
        writer.add_scalar("diag/late_ploss", d["late_ploss"], iteration)
        writer.add_scalar("diag/buffer_fill", d["buffer_fill"], iteration)
        writer.add_scalar("diag/pred_v_mean", d["pred_v_mean"], iteration)
        writer.add_scalar("diag/pred_v_std", d["pred_v_std"], iteration)
        writer.add_scalar("diag/pred_v_abs_mean", d["pred_v_abs_mean"], iteration)
        writer.add_scalar("diag/policy_grad_frac", d["policy_grad_frac"], iteration)
        writer.add_scalar("diag/val_loss_floor", d["val_loss_floor"], iteration)
        writer.add_scalar("diag/value_grad_norm", d["avg_value_grad_norm"], iteration)
        writer.add_scalar("diag/policy_grad_norm", d["avg_policy_grad_norm"], iteration)
        writer.add_scalar("diag/policy_entropy", d["policy_entropy"], iteration)
        writer.add_scalar("diag/policy_top1_acc", d["policy_top1_acc"], iteration)
        writer.add_scalar("diag/policy_top3_acc", d["policy_top3_acc"], iteration)
        writer.add_scalar("diag/value_confidence_acc", d["value_confidence_acc"], iteration)
        writer.add_scalar("diag/value_confident_frac", d["value_confident_frac"], iteration)
        for rb_i, rb_gn in d.get("rb_grad_norms", {}).items():
            writer.add_scalar(f"diag/rb{rb_i}_grad_norm", rb_gn, iteration)
        if d.get("vloss_delta") is not None:
            writer.add_scalar("diag/vloss_delta", d["vloss_delta"], iteration)
            writer.add_scalar("diag/pre_train_vloss", d["pre_train_vloss"], iteration)
        writer.add_scalar("diag/phase_vloss_early", d["phase_vloss_early"], iteration)
        writer.add_scalar("diag/phase_vloss_mid", d["phase_vloss_mid"], iteration)
        writer.add_scalar("diag/phase_vloss_late", d["phase_vloss_late"], iteration)
        writer.add_scalar("diag/policy_loss_decisive", d["policy_loss_decisive"], iteration)
        writer.add_scalar("diag/policy_loss_ambiguous", d["policy_loss_ambiguous"], iteration)
        writer.add_scalar("diag/decisive_frac", d["decisive_frac"], iteration)

        # Console output
        print(f"  Diag: targets mean={d['val_target_mean']:+.3f} "
              f"std={d['val_target_std']:.3f} | "
              f"X={d['frac_neg']:.1%} O={d['frac_pos']:.1%} "
              f"draw={d['frac_draw']:.1%}")
        overfit_gap = d['val_vloss'] - d['late_vloss']
        writer.add_scalar("diag/overfit_gap_vloss", overfit_gap, iteration)
        vloss_delta_str = f" delta={d['vloss_delta']:+.4f}" if d.get('vloss_delta') is not None else ""
        print(f"  Diag: eff_epochs={d['effective_epochs']:.1f} "
              f"vlw={d.get('effective_vlw',1.0):.2f} "
              f"steps={d['num_steps']} | "
              f"vloss train={d['late_vloss']:.4f} "
              f"val={d['val_vloss']:.4f} "
              f"(gap={overfit_gap:+.4f}){vloss_delta_str} | "
              f"buf={d['buffer_fill']}/{d['buffer_capacity']}"
              f"{' FULL' if d['buffer_full'] else ''}")
        writer.add_scalar("diag/val_vloss", d["val_vloss"], iteration)
        writer.add_scalar("diag/effective_vlw", d.get("effective_vlw", 1.0), iteration)
        writer.add_scalar("diag/val_ploss", d["val_ploss"], iteration)
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
        print(f"  Diag[P]: entropy={d['policy_entropy']:.3f} "
              f"top1_acc={d['policy_top1_acc']:.1%} "
              f"top3_acc={d['policy_top3_acc']:.1%}")
        print(f"  Diag[P2]: ploss_decisive={d['policy_loss_decisive']:.4f} "
              f"ploss_ambiguous={d['policy_loss_ambiguous']:.4f} "
              f"decisive_frac={d['decisive_frac']:.1%}")
        print(f"  Diag[C]: confident_acc={d['value_confidence_acc']:.1%} "
              f"(frac_confident={d['value_confident_frac']:.1%})")
        rb_gn = d.get('rb_grad_norms', {})
        if rb_gn:
            rb_items = sorted((i, n) for i, n in rb_gn.items() if isinstance(i, int))
            rb_str = " ".join(f"rb{i}={n:.4f}" for i, n in rb_items)
            eff_lr_items = sorted((k, v) for k, v in rb_gn.items() if isinstance(k, str) and 'eff_lr' in k)
            eff_lr_str = " ".join(f"rb{k.replace('_eff_lr','')}={v:.5f}" for k, v in eff_lr_items)
            line = f"  Diag[RB]: grad_norms: {rb_str}"
            if eff_lr_str:
                line += f" | c2_eff_lr: {eff_lr_str}"
            print(line)
        vh_bins = d.get('val_hist', [])
        if vh_bins:
            print(f"  Diag[TH]: targets [-1,-0.5)={vh_bins[0]:.1%} "
                  f"[-0.5,0)={vh_bins[1]:.1%} [0]={vh_bins[2]:.1%} "
                  f"(0,0.5]={vh_bins[3]:.1%} (0.5,1]={vh_bins[4]:.1%}")
        tr = d.get('three_r_diag', {})
        if tr:
            mine = tr.get('mine', {})
            opp = tr.get('opp', {})
            print(f"  Diag[3R]: ch0_3row: n={mine.get('count',0)} target={mine.get('mean_target',0):+.3f} "
                  f"({mine.get('frac',0):.1%}) | "
                  f"ch1_3row: n={opp.get('count',0)} target={opp.get('mean_target',0):+.3f} "
                  f"({opp.get('frac',0):.1%})")
            writer.add_scalar("diag/three_r_ch0_target", mine.get('mean_target', 0), iteration)
            writer.add_scalar("diag/three_r_ch1_target", opp.get('mean_target', 0), iteration)
            writer.add_scalar("diag/three_r_ch0_frac", mine.get('frac', 0), iteration)
            writer.add_scalar("diag/three_r_ch1_frac", opp.get('frac', 0), iteration)
        print(f"  Diag[A]: X_vloss={d['x_vloss']:.4f} O_vloss={d['o_vloss']:.4f} | "
              f"X_target={d['x_target_mean']:+.3f} O_target={d['o_target_mean']:+.3f} | "
              f"X_pred={d['x_pred_mean']:+.3f} O_pred={d['o_pred_mean']:+.3f}")
        print(f"  Diag[BufBias]: X-to-move: n={d.get('buf_n_x',0)} "
              f"mean_tgt={d.get('buf_mean_tgt_x',0):+.3f} "
              f"frac_pos={d.get('buf_frac_pos_x',0):.1%} | "
              f"O-to-move: n={d.get('buf_n_o',0)} "
              f"mean_tgt={d.get('buf_mean_tgt_o',0):+.3f} "
              f"frac_pos={d.get('buf_frac_pos_o',0):.1%}")
        writer.add_scalar("pbias/buf_mean_tgt_x", d.get('buf_mean_tgt_x', 0), iteration)
        writer.add_scalar("pbias/buf_mean_tgt_o", d.get('buf_mean_tgt_o', 0), iteration)
        writer.add_scalar("pbias/buf_frac_pos_x", d.get('buf_frac_pos_x', 0), iteration)
        writer.add_scalar("pbias/buf_frac_pos_o", d.get('buf_frac_pos_o', 0), iteration)
        print(f"  Diag[PBias]: pre: X_pred={d.get('pbias_pre_x_pred',0):+.3f} "
              f"X_acc={d.get('pbias_pre_x_acc',0):.1%} | "
              f"O_pred={d.get('pbias_pre_o_pred',0):+.3f} "
              f"O_acc={d.get('pbias_pre_o_acc',0):.1%}")
        print(f"  Diag[PBias]: post: X_pred={d.get('pbias_post_x_pred',0):+.3f} "
              f"X_acc={d.get('pbias_post_x_acc',0):.1%} | "
              f"O_pred={d.get('pbias_post_o_pred',0):+.3f} "
              f"O_acc={d.get('pbias_post_o_acc',0):.1%}")
        writer.add_scalar("pbias/pre_x_pred", d.get('pbias_pre_x_pred', 0), iteration)
        writer.add_scalar("pbias/pre_o_pred", d.get('pbias_pre_o_pred', 0), iteration)
        writer.add_scalar("pbias/post_x_pred", d.get('pbias_post_x_pred', 0), iteration)
        writer.add_scalar("pbias/post_o_pred", d.get('pbias_post_o_pred', 0), iteration)
        writer.add_scalar("pbias/pre_x_acc", d.get('pbias_pre_x_acc', 0), iteration)
        writer.add_scalar("pbias/pre_o_acc", d.get('pbias_pre_o_acc', 0), iteration)
        writer.add_scalar("pbias/post_x_acc", d.get('pbias_post_x_acc', 0), iteration)
        writer.add_scalar("pbias/post_o_acc", d.get('pbias_post_o_acc', 0), iteration)
        pc = d.get('phase_counts', {})
        print(f"  Diag[GP]: vloss early={d['phase_vloss_early']:.4f}({pc.get('early',0)}) "
              f"mid={d['phase_vloss_mid']:.4f}({pc.get('mid',0)}) "
              f"late={d['phase_vloss_late']:.4f}({pc.get('late',0)})")
        gs = d.get('grad_stats', {})
        if gs:
            err_trend = gs.get('error_mean_trend', [])
            trend_str = ""
            if len(err_trend) >= 2:
                trend_str = f" err_trend={err_trend[0]:+.3f}->{err_trend[-1]:+.3f}"
            print(f"  Diag[F]: fc1_grad={gs.get('fc1_grad_norm_mean',0):.4f} "
                  f"fc2_grad={gs.get('fc2_grad_norm_mean',0):.4f} | "
                  f"fc1_mean={gs.get('fc1_grad_mean',0):+.6f} "
                  f"fc2_mean={gs.get('fc2_grad_mean',0):+.6f}"
                  f"{trend_str}")

        vh = d.get('vh_diag', {})
        if vh:
            self._log_value_head_diagnostics(vh, d, iteration)

        # Gradient conflict diagnostic
        gc = d.get('grad_conflict', {})
        if gc:
            print(f"  Diag[GCONF]: x_vs_o grad cosine: "
                  f"all={gc['cos_all']:+.3f} "
                  f"backbone={gc['cos_backbone']:+.3f} "
                  f"value_head={gc['cos_value_head']:+.3f} | "
                  f"x_pred={gc['x_pred_scalar']:+.3f} "
                  f"o_pred={gc['o_pred_scalar']:+.3f}")
            writer.add_scalar("grad_conflict/cos_all", gc['cos_all'], iteration)
            writer.add_scalar("grad_conflict/cos_backbone", gc['cos_backbone'], iteration)
            writer.add_scalar("grad_conflict/cos_value_head", gc['cos_value_head'], iteration)
            writer.add_scalar("grad_conflict/x_pred", gc['x_pred_scalar'], iteration)
            writer.add_scalar("grad_conflict/o_pred", gc['o_pred_scalar'], iteration)

        # ImmWin vloss split
        imm_vloss = d.get('imm_win_vloss', 0)
        non_imm_vloss = d.get('non_imm_win_vloss', 0)
        imm_frac = d.get('imm_win_frac_train', 0)
        if imm_vloss > 0 or non_imm_vloss > 0:
            print(f"  Diag[IMM]: vloss imm_win={imm_vloss:.4f} "
                  f"non_imm={non_imm_vloss:.4f} "
                  f"ratio={imm_vloss / max(non_imm_vloss, 1e-8):.2f} "
                  f"imm_frac={imm_frac:.1%}")
            writer.add_scalar("diag/imm_win_vloss", imm_vloss, iteration)
            writer.add_scalar("diag/non_imm_win_vloss", non_imm_vloss, iteration)
            writer.add_scalar("diag/imm_win_frac_train", imm_frac, iteration)

    def _log_vh_tensorboard(self, vh, iteration):
        """Write all value head diagnostics to TensorBoard using data-driven tables."""
        writer = self.writer
        for tb_tag, vh_key in self._VH_SCALARS:
            if vh_key in vh:
                writer.add_scalar(tb_tag, vh[vh_key], iteration)
        for i, n in enumerate(vh.get('pconv_ch_norms', [])):
            writer.add_scalar(f"ph/pconv_ch{i}_norm", n, iteration)
        for dict_key, mappings in self._VH_PER_BLOCK:
            for bi, block_data in vh.get(dict_key, {}).items():
                for tb_suffix, subkey in mappings:
                    if subkey in block_data:
                        writer.add_scalar(f"rb{bi}/{tb_suffix}", block_data[subkey], iteration)
        for bi, nv in vh.get('rb_ch_dominance', {}).items():
            writer.add_scalar(f"rb{bi}/ch_value_dom", nv, iteration)
        for bn, sv in vh.get('rb_v_grad_survival', {}).items():
            writer.add_scalar(f"rb{bn}/v_grad_survival", sv, iteration)
        for bi, ratio in vh.get('rb_residual_ratios', {}).items():
            writer.add_scalar(f"rb{bi}/residual_ratio", ratio, iteration)

    def _log_value_head_diagnostics(self, vh, d, iteration):
        """Log value head health diagnostics to console and TensorBoard."""
        print(f"  Diag[V]: dead={vh['dead_neurons']} "
              f"active={vh.get('active_neurons','?')}/{vh['total_neurons']} "
              f"| WDL: entropy={vh['wdl_entropy']:.3f} "
              f"conf={vh['wdl_confidence']:.3f} "
              f"acc={vh['wdl_accuracy']:.1%} "
              f"v={vh['wdl_scalar_mean']:+.3f}±{vh['wdl_scalar_std']:.3f}")
        print(f"  Diag[V1]: WDL probs: W={vh['wdl_win_prob']:.3f} "
              f"D={vh['wdl_draw_prob']:.3f} L={vh['wdl_loss_prob']:.3f} | "
              f"logit_std={vh['wdl_logit_std']:.3f} "
              f"logit_range={vh['wdl_logit_range']:.3f}")
        print(f"  Diag[V2]: fc2_w=[{vh['fc2_w_min']:+.3f},{vh['fc2_w_max']:+.3f}] "
              f"norm={vh['fc2_w_norm']:.3f} "
              f"bias=[W:{vh['fc2_bias_w']:+.3f} D:{vh['fc2_bias_d']:+.3f} L:{vh['fc2_bias_l']:+.3f}] | "
              f"fc1_w_norm={vh['fc1_w_norm']:.3f} | "
              f"backbone std={vh['backbone_std']:.3f} |x|={vh['backbone_abs_mean']:.3f}")
        print(f"  Diag[V4]: fc1_act p10={vh['fc1_act_p10']:.4f} "
              f"p50={vh['fc1_act_p50']:.4f} "
              f"p90={vh['fc1_act_p90']:.4f}")
        ch_str = " ".join(f"ch{i}={vh['vconv_ch_abs_mean'][i]:.3f}"
                          for i in range(vh['vconv_n_channels']))
        print(f"  Diag[V5]: vconv_channels: {ch_str} "
              f"dead_ch={vh['vconv_dead_channels']}/{vh['vconv_n_channels']}")
        dead_ids = vh.get('dead_neuron_ids', [])
        weak = list(zip(vh['weakest_5_ids'], vh['weakest_5_vals']))
        print(f"  Diag[V6]: dead_ids={dead_ids[:10]}"
              f"{'...' if len(dead_ids) > 10 else ''} | "
              f"weakest={weak}")
        print(f"  Diag[V7]: fc2_w_d={vh['fc2_w_norm_delta']:+.4f} "
              f"fc1_w_d={vh['fc1_w_norm_delta']:+.4f}")
        print(f"  Diag[V8]: grad_flow "
              f"dead={vh['grad_dead_mean']:.6f} "
              f"alive={vh['grad_alive_mean']:.6f} "
              f"ratio={vh['grad_dead_mean']/max(vh['grad_alive_mean'],1e-10):.3f}")
        print(f"  Diag[VC]: backbone_raw |x|={vh['backbone_raw_abs']:.3f} "
              f"std={vh['backbone_raw_std']:.3f} | "
              f"vconv_w norm={vh['vc_w_norm']:.3f} |w|={vh['vc_w_abs_mean']:.4f}")
        print(f"  Diag[VC2]: pre_bn |x|={vh['vconv_pre_bn_abs']:.3f} "
              f"std={vh['vconv_pre_bn_std']:.3f} | "
              f"post_bn |x|={vh['vbn_post_abs']:.3f} "
              f"std={vh['vbn_post_std']:.3f} | "
              f"bn_ratio={vh['bn_ratio']:.3f}")
        gamma = vh['vbn_gamma']
        beta = vh['vbn_beta']
        gamma_str = " ".join(f"{g:.3f}" for g in gamma)
        beta_str = " ".join(f"{b:+.3f}" for b in beta)
        print(f"  Diag[VC3]: gn_gamma=[{gamma_str}] "
              f"min={vh['vbn_gamma_min']:.3f}")
        print(f"  Diag[VC4]: gn_beta=[{beta_str}]")
        print(f"  Diag[VC6]: policy_conv |x|={vh['pconv_abs']:.3f} "
              f"std={vh['pconv_std']:.3f} | "
              f"vconv_grad={vh['vconv_grad_norm']:.4f}")
        if 'bb_n_channels' in vh:
            print(f"  Diag[BB]: backbone {vh['bb_n_channels']}ch "
                  f"dead={vh['bb_dead_channels']} | "
                  f"|x| p10={vh['bb_ch_p10']:.4f} "
                  f"p50={vh['bb_ch_p50']:.4f} "
                  f"p90={vh['bb_ch_p90']:.4f} "
                  f"max={vh['bb_ch_max']:.4f}")
            print(f"  Diag[BB1]: top5={vh['bb_top5']} "
                  f"bot5={vh['bb_bot5']}")
        if 'bb_v_grad_norm' in vh:
            evlw = d.get('effective_vlw', 1.0)
            v_eff = vh['bb_v_grad_norm'] * evlw
            eff_ratio = v_eff / max(vh['bb_p_grad_norm'], 1e-10)
            print(f"  Diag[BB2]: bb_grad output: "
                  f"v={vh['bb_v_grad_norm']:.4f} "
                  f"p={vh['bb_p_grad_norm']:.4f} "
                  f"ratio={vh['bb_grad_ratio']:.3f} "
                  f"(eff_v={v_eff:.4f} eff_ratio={eff_ratio:.3f})")
            print(f"  Diag[BB3]: ch_dominance: "
                  f"value={vh['bb_n_value_dom']}/{vh.get('bb_n_channels',256)} "
                  f"policy={vh['bb_n_policy_dom']}/{vh.get('bb_n_channels',256)} | "
                  f"top_v={vh['bb_top_v_channels'][:3]} "
                  f"top_p={vh['bb_top_p_channels'][:3]}")
        if 'bb_param_v_grad' in vh:
            evlw = d.get('effective_vlw', 1.0)
            pv_eff = vh['bb_param_v_grad'] * evlw
            pv_eff_ratio = pv_eff / max(vh['bb_param_p_grad'], 1e-10)
            rb = vh.get('bb_res_block_grads', {})
            rb_str = " ".join(
                f"rb{k}: v={v[0]:.3f} p={v[1]:.3f} r={v[2]:.2f}"
                for k, v in sorted(rb.items()))
            print(f"  Diag[BB4]: bb_grad params: "
                  f"v={vh['bb_param_v_grad']:.4f} "
                  f"p={vh['bb_param_p_grad']:.4f} "
                  f"ratio={vh['bb_param_grad_ratio']:.3f} "
                  f"(eff_v={pv_eff:.4f} eff_ratio={pv_eff_ratio:.3f})")
            if rb_str:
                print(f"  Diag[BB5]: {rb_str}")
            vgs = vh.get('rb_v_grad_survival', {})
            if vgs:
                vgs_str = " ".join(f"rb{k}={v:.3f}" for k, v in sorted(vgs.items()))
                print(f"  Diag[BB7]: v_grad_survival: {vgs_str}")
        if 'bb_grad_cosine_sim' in vh:
            print(f"  Diag[BB6]: grad_cosine={vh['bb_grad_cosine_sim']:+.3f} "
                  f"conflict_ch={vh['bb_grad_conflict_channels']} "
                  f"aligned_ch={vh['bb_grad_aligned_channels']}")
        if 'vp_weight_corr' in vh:
            print(f"  Diag[VP]: weight_corr={vh['vp_weight_corr']:.3f} "
                  f"overlap_top20={vh['vp_overlap_20']}/20 | "
                  f"act: val_top20={vh['vp_val_top20_act']:.4f} "
                  f"pol_top20={vh['vp_pol_top20_act']:.4f} "
                  f"ratio={vh['vp_val_top20_act']/max(vh['vp_pol_top20_act'],1e-10):.2f}")
            print(f"  Diag[VP2]: health_corr: "
                  f"val={vh['vp_val_health_corr']:.3f} "
                  f"pol={vh['vp_pol_health_corr']:.3f}")
        if 'svd_bb_rank90' in vh:
            print(f"  Diag[SVD]: backbone rb{len(self._t.net.res_blocks)-1}.conv2: "
                  f"rank90={vh['svd_bb_rank90']}/{vh['svd_bb_total']} "
                  f"rank99={vh['svd_bb_rank99']}/{vh['svd_bb_total']} "
                  f"near_zero_sv={vh['svd_bb_near_zero']} "
                  f"gn_dead={vh['gn_dead_deepest']}")
            pc_str = " ".join(f"ch{i}={n:.3f}" for i, n in enumerate(vh['pconv_ch_norms']))
            print(f"  Diag[PH]: policy_fc: "
                  f"rank90={vh['svd_pfc_rank90']}/{vh['svd_pfc_max_rank']} "
                  f"rank99={vh['svd_pfc_rank99']}/{vh['svd_pfc_max_rank']} | "
                  f"pconv: {pc_str}")
        all_rb_bn_data = vh.get('all_rb_bn', {})
        rb_act_data = vh.get('rb_act_stats', {})
        for bi in sorted(set(list(all_rb_bn_data.keys()) + list(rb_act_data.keys()))):
            parts = [f"  Diag[RB{bi}]:"]
            if bi in all_rb_bn_data:
                rbd = all_rb_bn_data[bi]
                parts.append(f"gn2: dead={rbd.get('dead',0)} "
                             f"neg_gamma={rbd.get('neg_gamma',0)} "
                             f"gamma={rbd.get('gamma_mean',0):.3f}(+/-{rbd.get('gamma_std',0):.3f})")
                parts.append(f"svd_rank90={rbd.get('svd_rank90',0)}/"
                             f"{rbd.get('svd_total',0)}")
            if bi in rb_act_data:
                rad = rb_act_data[bi]
                parts.append(f"|x|={rad['abs_mean']:.3f} "
                             f"std={rad['std']:.3f} "
                             f"dead={rad['dead_channels']}")
            print(" | ".join(parts))
            parts2 = []
            rcn = vh.get('rb_conv_norms', {}).get(bi)
            if rcn:
                parts2.append(f"w: c1={rcn['conv1']:.3f} c2={rcn['conv2']:.3f} c2_d={rcn.get('c2_delta',0):+.3f}")
            rbs = vh.get('rb_bn2_stats', {}).get(bi)
            if rbs:
                s = ""
                if 'bn2_out_abs' in rbs:
                    s += f"bn2_out: |x|={rbs['bn2_out_abs']:.3f} std={rbs['bn2_out_std']:.3f}"
                if 'conv2_raw_var' in rbs:
                    if s: s += " | "
                    s += f"conv2_raw: var={rbs['conv2_raw_var']:.4f} |x|={rbs['conv2_raw_abs']:.3f}"
                if 'bn2_batch_var_mean' in rbs:
                    if s: s += " | "
                    s += f"batch_var={rbs['bn2_batch_var_mean']:.4f}"
                if s:
                    parts2.append(s)
            rrk = vh.get('rb_res_rank', {}).get(bi)
            if rrk:
                parts2.append(f"res_rank90={rrk['rank90']}/{rrk['total']}")
            rcd = vh.get('rb_ch_dominance', {})
            if bi in rcd:
                parts2.append(f"ch_vdom={rcd[bi]}/{vh.get('bb_n_channels', '?')}")
            if parts2:
                print(f"           {' | '.join(parts2)}")
        if 'final_bn_gamma_mean' in vh:
            print(f"  Diag[FGN]: final_gn "
                  f"gamma={vh.get('final_bn_gamma_mean',0):.3f}(+/-{vh.get('final_bn_gamma_std',0):.3f}) "
                  f"dead={vh['final_bn_dead']}")
        rr = vh.get('rb_residual_ratios', {})
        if rr:
            rr_str = " ".join(f"rb{k}={v:.3f}" for k, v in sorted(rr.items()))
            print(f"  Diag[RR]: residual_ratio: {rr_str}")
        if 'init_conv_w_norm' in vh:
            print(f"  Diag[IC]: conv_w norm={vh['init_conv_w_norm']:.4f} "
                  f"|w|={vh['init_conv_w_abs_mean']:.5f} "
                  f"delta={vh['init_conv_w_norm_delta']:+.4f}")

        self._log_vh_tensorboard(vh, iteration)

    def _log_selfplay_value_diagnostics(self, iteration):
        """Log self-play value prediction diagnostics."""
        t = self._t
        writer = self.writer
        if hasattr(t, '_batched') and hasattr(t._batched, 'value_diag'):
            vd = t._batched.value_diag
            if vd:
                writer.add_scalar("selfplay_diag/mean_nnet_value", vd["mean_nnet_value"], iteration)
                writer.add_scalar("selfplay_diag/std_nnet_value", vd["std_nnet_value"], iteration)
                writer.add_scalar("selfplay_diag/frac_saturated", vd["frac_saturated_any"], iteration)
                writer.add_scalar("selfplay_diag/sign_accuracy", vd["sign_accuracy"], iteration)
                writer.add_scalar("selfplay_diag/mae_vs_outcome", vd["mae_vs_outcome"], iteration)
                writer.add_scalar("selfplay_diag/pred_outcome_corr", vd["pred_outcome_corr"], iteration)
                writer.add_scalar("selfplay_diag/mean_when_x_moves", vd["mean_when_x_moves"], iteration)
                writer.add_scalar("selfplay_diag/mean_when_o_moves", vd["mean_when_o_moves"], iteration)
                print(f"  SelfPlay: nnet_v mean={vd['mean_nnet_value']:+.3f} "
                      f"std={vd['std_nnet_value']:.3f} | "
                      f"hi_conf={vd['frac_saturated_any']:.1%} | "
                      f"sign_acc={vd['sign_accuracy']:.1%} | "
                      f"MAE={vd['mae_vs_outcome']:.3f} | "
                      f"corr={vd['pred_outcome_corr']:+.3f}")
                print(f"  SelfPlay: v_when_X={vd['mean_when_x_moves']:+.3f} "
                      f"v_when_O={vd['mean_when_o_moves']:+.3f} | "
                      f"conf+={vd['frac_saturated_pos']:.1%} "
                      f"conf-={vd['frac_saturated_neg']:.1%}")
                if 'mcts_visit_entropy_mean' in vd:
                    writer.add_scalar("selfplay_diag/mcts_visit_entropy",
                                       vd["mcts_visit_entropy_mean"], iteration)
                    print(f"  SelfPlay: mcts_visit_entropy="
                          f"{vd['mcts_visit_entropy_mean']:.3f} "
                          f"(std={vd['mcts_visit_entropy_std']:.3f})")
                if 'mcts_nnet_corr' in vd:
                    writer.add_scalar("selfplay_diag/mcts_nnet_corr", vd["mcts_nnet_corr"], iteration)
                    writer.add_scalar("selfplay_diag/mcts_nnet_mae", vd["mcts_nnet_mae"], iteration)
                    writer.add_scalar("selfplay_diag/mcts_correction_mean", vd["mcts_correction_mean"], iteration)
                    print(f"  SelfPlay: mcts_Q mean={vd['mcts_q_mean']:+.3f} "
                          f"std={vd['mcts_q_std']:.3f} | "
                          f"nnet_Q_corr={vd['mcts_nnet_corr']:+.3f} "
                          f"MAE={vd['mcts_nnet_mae']:.3f} "
                          f"correction={vd['mcts_correction_mean']:+.3f}")

    def _log_intra_iteration_dynamics(self, iteration):
        """Log confidence distribution, sub-iteration dynamics, and value trajectory."""
        t = self._t
        writer = self.writer
        if hasattr(t, '_train_diag'):
            cd = t._train_diag.get('conf_dist', {})
            if cd:
                print(f"  Diag[CONF]: |v|<0.1={cd.get('very_low',0):.1%} "
                      f"0.1-0.3={cd.get('low',0):.1%} "
                      f"0.3-0.6={cd.get('medium',0):.1%} "
                      f"0.6-0.9={cd.get('high',0):.1%} "
                      f"|v|>0.9={cd.get('very_high',0):.1%}")
                for bk, bv in cd.items():
                    writer.add_scalar(f"conf/{bk}", bv, iteration)

            sil = t._train_diag.get('sub_iter_log', [])
            if len(sil) >= 3:
                indices = [0, len(sil) // 2, len(sil) - 1]
                parts = []
                for idx in indices:
                    e = sil[idx]
                    parts.append(f"s{e['step']}: vl={e['vloss']:.4f} "
                                 f"pl={e['ploss']:.4f} |v|={e['mean_conf']:.3f} "
                                 f"v={e['mean_v']:+.3f}")
                print(f"  Diag[SUB]: {' -> '.join(parts)}")
                writer.add_scalar("sub/vloss_first", sil[0]['vloss'], iteration)
                writer.add_scalar("sub/vloss_last", sil[-1]['vloss'], iteration)
                writer.add_scalar("sub/conf_first", sil[0]['mean_conf'], iteration)
                writer.add_scalar("sub/conf_last", sil[-1]['mean_conf'], iteration)
                writer.add_scalar("sub/mean_v_first", sil[0]['mean_v'], iteration)
                writer.add_scalar("sub/mean_v_last", sil[-1]['mean_v'], iteration)

        if hasattr(t, '_train_diag'):
            _traj = t._train_diag.get('fixed_eval_trajectory', [])
            if _traj and len(_traj) >= 2:
                _names = [k for k in _traj[0] if k != 'step']
                for _pn in _names:
                    _vals = [f"s{e['step']}:{e[_pn]:+.3f}" for e in _traj]
                    print(f"  Diag[TRAJ]: {_pn}: {' -> '.join(_vals)}")
                    writer.add_scalar(f"traj/{_pn}_first", _traj[0][_pn], iteration)
                    writer.add_scalar(f"traj/{_pn}_last", _traj[-1][_pn], iteration)

            # Swap-representation cosine similarity
            _swap_repr = t._train_diag.get('swap_repr_trajectory', [])
            if _swap_repr and len(_swap_repr) >= 1:
                _sr_names = [k for k in _swap_repr[0] if k != 'step']
                _sr_parts = []
                for _pn in _sr_names:
                    _sr_parts.append(f"{_pn}={_swap_repr[-1][_pn]:+.3f}")
                    writer.add_scalar(f"swap_repr/{_pn}_cosine", _swap_repr[-1][_pn], iteration)
                last = _swap_repr[-1]
                mean_cos = np.mean([last[k] for k in _sr_names])
                writer.add_scalar("swap_repr/mean_cosine", mean_cos, iteration)
                print(f"  Diag[SWAP_REPR]: {' '.join(_sr_parts)} | mean={mean_cos:+.3f}")

            # Value sensitivity
            _fe_sens = t._train_diag.get('fe_sensitivity', {})
            if _fe_sens:
                parts = []
                for name, s in _fe_sens.items():
                    parts.append(f"{name}: d={s['mean_abs_delta']:.3f}/{s['max_abs_delta']:.3f} "
                                 f"drift={s['total_drift']:.3f}")
                    writer.add_scalar(f"fe_sens/{name}_avg_delta", s['mean_abs_delta'], iteration)
                    writer.add_scalar(f"fe_sens/{name}_max_delta", s['max_abs_delta'], iteration)
                    writer.add_scalar(f"fe_sens/{name}_total_drift", s['total_drift'], iteration)
                print(f"  Diag[SENS]: {' | '.join(parts)}")

            # GN sanity: train/eval gap (first iter only)
            _gn_sanity = t._train_diag.get('gn_sanity', {})
            if _gn_sanity:
                _gs_parts = [f"{k}={v:.6f}" for k, v in _gn_sanity['per_pos'].items()]
                _status = "OK" if _gn_sanity['max_gap'] < 1e-6 else "FAIL"
                print(f"  Diag[GN_SANITY]: {_status} max_gap={_gn_sanity['max_gap']:.2e} "
                      f"| {' '.join(_gs_parts)}")
                writer.add_scalar("gn_sanity/max_gap", _gn_sanity['max_gap'], iteration)

            # GN health: group variance, saturation, gamma
            _gn_gvar = t._train_diag.get('gn_group_var_trajectory', [])
            if _gn_gvar:
                _gv_last = _gn_gvar[-1]
                _warnings = []
                if _gv_last['num_degenerate'] > 0:
                    _warnings.append(f"DEGENERATE:{_gv_last['num_degenerate']}/{_gv_last['num_layers']}")
                if _gv_last.get('sat_frac', 0) > 0.05:
                    _warnings.append(f"SAT:{_gv_last['sat_frac']:.1%}")
                if _gv_last.get('gamma_max', 0) > 5.0:
                    _warnings.append(f"GAMMA_HIGH:{_gv_last['gamma_max']:.1f}")
                if _gv_last.get('gamma_min', 1) < 0.01:
                    _warnings.append(f"GAMMA_LOW:{_gv_last['gamma_min']:.3f}")
                _warn_str = f" | {' '.join(_warnings)}" if _warnings else ""
                print(f"  Diag[GN_HEALTH]: var={_gv_last['min_var']:.2e}/{_gv_last['mean_var']:.4f} "
                      f"sat={_gv_last.get('sat_frac', 0):.3f} "
                      f"gamma=[{_gv_last.get('gamma_min', 0):.3f},{_gv_last.get('gamma_max', 0):.3f}]"
                      f"{_warn_str}")
                writer.add_scalar("gn_health/min_var", _gv_last['min_var'], iteration)
                writer.add_scalar("gn_health/mean_var", _gv_last['mean_var'], iteration)
                writer.add_scalar("gn_health/num_degenerate", _gv_last['num_degenerate'], iteration)
                writer.add_scalar("gn_health/sat_frac", _gv_last.get('sat_frac', 0), iteration)
                writer.add_scalar("gn_health/gamma_max", _gv_last.get('gamma_max', 0), iteration)
                writer.add_scalar("gn_health/gamma_min", _gv_last.get('gamma_min', 0), iteration)

    def eval_diagnostic_positions(self, iteration, prefix="", label="FixedEval"):
        """Evaluate the network on fixed diagnostic positions every iteration."""
        t = self._t
        t.net.eval()
        positions = self._get_diagnostic_positions()
        if not positions:
            return

        if not getattr(self, '_encoding_checked', False):
            self._encoding_checked = True
            print(f"  Diag[ENC]: Fixed position encoding check (canonical, {positions[0][1].shape[0]}ch):")
            for name, state_input, _ in positions:
                # Infer player from piece counts: equal = X to move
                my_count = state_input[0].sum()
                opp_count = state_input[1].sum()
                player_str = "X" if my_count == opp_count else "O"
                ch0_rows = []
                ch1_rows = []
                for r in range(state_input.shape[1]):
                    ch0_rows.append("".join(
                        "1" if state_input[0, r, c] > 0 else "."
                        for c in range(state_input.shape[2])))
                    ch1_rows.append("".join(
                        "1" if state_input[1, r, c] > 0 else "."
                        for c in range(state_input.shape[2])))
                print(f"    {name}: player={player_str} "
                      f"ch0(me)={'/'.join(ch0_rows)} "
                      f"ch1(opp)={'/'.join(ch1_rows)}")

        # Build all 15 inputs (original + swapped + mirrored) for a single
        # batched forward pass instead of 15 individual predict() calls.
        # With canonical encoding (no ch2), swapped and mirrored are identical
        # (both just swap ch0↔ch1 to view from opponent's perspective).
        originals = []
        swapped = []
        mirrored = []
        for _, state_input, _ in positions:
            originals.append(state_input)
            sw = state_input.copy()
            sw[0], sw[1] = state_input[1].copy(), state_input[0].copy()
            swapped.append(sw)
            mirrored.append(sw.copy())  # same as swapped with canonical encoding

        all_inputs = originals + swapped + mirrored
        all_values, all_policies = t.net.batch_predict(all_inputs)

        n = len(positions)
        print(f"  {label}:")
        for i, (name, state_input, expected_str) in enumerate(positions):
            value = all_values[i]
            policy = all_policies[i]
            top_action = np.argmax(policy)
            swap_delta = value - all_values[n + i]
            sym_err = value + all_values[2 * n + i]
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_value", value, iteration)
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_top_action", top_action, iteration)
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_swap_delta", swap_delta, iteration)
            self.writer.add_scalar(f"fixed_eval/{prefix}{name}_sym_err", sym_err, iteration)
            print(f"    {name}: V={value:+.4f} top_act={top_action} swap_d={swap_delta:+.4f} "
                  f"sym={sym_err:+.4f} ({expected_str})")

    def _get_diagnostic_positions(self):
        """Return a list of (name, state_input, expected_description) for fixed evaluation."""
        t = self._t
        positions = []
        try:
            from games.connect4 import Connect4Game, GameState as C4State
            if not isinstance(t.game, Connect4Game):
                return positions
        except ImportError:
            return positions

        s = t.game.new_game()
        positions.append(("empty_board", t.game.state_to_input(s), "expect ~0"))

        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = -1
        board[0][4] = 1
        board[0][5] = 1
        board[1][4] = 1
        s = C4State(None, board, player=-1)
        positions.append(("x_wins_next", t.game.state_to_input(s), "expect > +0.5 (I'm winning)"))

        board = np.zeros((6, 7), dtype="int")
        board[0][0:3] = 1
        board[0][4:7] = -1
        board[1][4] = -1
        s = C4State(None, board, player=1)
        positions.append(("o_wins_next", t.game.state_to_input(s), "expect > +0.5 (I'm winning)"))

        board = np.zeros((6, 7), dtype="int")
        board[0] = [0, 1, 1, -1, 0, 1, -1]
        board[1] = [0, -1, 1, -1, 0, 0, 0]
        board[2] = [0, 1, -1, 1, 0, 0, 0]
        board[3] = [0, -1, 0, -1, 0, 0, 0]
        s = C4State(None, board, player=1)
        positions.append(("diag_threat", t.game.state_to_input(s), "expect < 0 (I'm losing)"))

        board = np.zeros((6, 7), dtype="int")
        board[0][3] = -1
        board[1][3] = -1
        board[0][2] = 1
        board[0][4] = 1
        s = C4State(None, board, player=-1)
        positions.append(("x_center", t.game.state_to_input(s), "expect > 0 (I'm slightly winning)"))

        return positions

