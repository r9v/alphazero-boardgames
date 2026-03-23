GAME_CONFIGS = {
    "tictactoe": {
        # --- Network ---
        "num_filters": 16,
        "num_res_blocks": 1,
        "value_head_channels": 1,
        "value_head_fc_size": 16,
        "policy_head_channels": 2,

        # --- Training ---
        "max_train_steps": 800,
        "target_epochs": 4,
        "buffer_size": 2_000,
        "value_loss_weight": 1.0,
        "lr": 0.01,
        "batch_size": 64,

        # --- Self-play ---
        "default_iterations": 24,
        "default_games": 24,
        "default_simulations": 24,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 3,
        "c_puct": 1.5,
        "dirichlet_alpha": 1.1,        # ~10 / 9 legal moves

        # --- Tree reuse ---
        "tree_reuse": True,

        # --- Play (human vs AI) ---
        "play_simulations": 16,
        "play_c_puct": 1.5,
    },

    "connect4": {
        # --- Network ---
        "num_filters": 64,
        "num_res_blocks": 4,
        "value_head_channels": 32,
        "value_head_fc_size": 64,
        "policy_head_channels": 2,
        # --- Training ---
        "train_ratio": 8,
        "value_loss_weight": 1.0,
        "value_label_smoothing": 0.1,
        "stv_weight": 0.0,             # short-term value target (MCTS Q) loss weight
        "surprise_kl_frac": 0.5,       # policy surprise: KL-proportional fraction (rest is uniform)
        "resblock_dropout": 0.1,
        "random_opening_moves": 12,
        "random_opening_fraction": 0.5,
        "buffer_size": 100_000,
        "lr": 0.005,
        "batch_size": 256,

        # --- Self-play ---
        "default_iterations": 64,
        "default_games": 256,
        "default_simulations": 200,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 25,
        "c_puct": 2.5,
        "dirichlet_alpha": 1.4,        # ~10 / 7 legal moves
        "dirichlet_epsilon": 0.4,      # 40% noise (was 0.25 default)

        # --- Search-contempt: Thompson sampling at opponent nodes after N visits ---
        "contempt_n": 20,  # 0=disabled, >0=Thompson after N visits at opp nodes

        # --- Tree reuse ---
        "tree_reuse": True,

        # --- Play (human vs AI) ---
        "play_simulations": 200,
        "play_c_puct": 2.5,
    },

    "santorini": {
        # --- Network ---
        "num_filters": 128,
        "num_res_blocks": 3,
        "value_head_channels": 4,
        "value_head_fc_size": 64,
        "policy_head_channels": 4,

        # --- Training ---
        "max_train_steps": 12800,
        "target_epochs": 4,
        "buffer_size": 55000,
        "value_loss_weight": 3.0,
        "lr": 0.01,
        "batch_size": 64,

        # --- Self-play ---
        "default_iterations": 64,
        "default_games": 64,
        "default_simulations": 256,
        "selects_per_round": 8,
        "vl_value": 3.0,
        "temp_threshold": 15,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.25,       # ~10 / ~40 avg legal moves

        # --- Tree reuse & Resign ---
        "tree_reuse": True,
        "resign_threshold": -0.99,
        "resign_min_moves": 10,
        "resign_check_prob": 0.1,

        # --- Play (human vs AI) ---
        "play_simulations": 400,
        "play_c_puct": 2.0,
    },
}
