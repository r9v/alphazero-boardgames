GAME_CONFIGS = {
    "tictactoe": {
        # Network
        "num_filters": 16,
        "num_res_blocks": 1,
        "num_groups": 2,
        "value_head_channels": 1,
        "value_head_fc_size": 16,
        "policy_head_channels": 2,

        # Training
        "max_train_steps": 800,
        "target_epochs": 4,
        "buffer_size": 2_000,
        "value_loss_weight": 1.0,
        "lr": 0.005,
        "batch_size": 64,

        # Self-play
        "default_iterations": 48,
        "default_games": 32,
        "default_simulations": 32,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 3,
        "c_puct": 1.5,
        "dirichlet_alpha": 1.1,   # ~10 / avg_legal_moves
        "tree_reuse": True,
        "play_simulations": 16,
        "play_c_puct": 1.5,
    },

    "connect4": {
        # Network
        "num_filters": 64,
        "num_res_blocks": 4,
        "value_head_channels": 32,
        "value_head_fc_size": 64,
        "policy_head_channels": 2,

        # Training
        "train_ratio": 8,
        "value_loss_weight": 1.0,
        "value_label_smoothing": 0.1,
        "surprise_kl_frac": 0.5,
        "resblock_dropout": 0.1,
        "random_opening_moves": 12,
        "random_opening_fraction": 0.5,
        "buffer_size": 100_000,
        "lr": 0.005,
        "batch_size": 256,

        # Self-play
        "default_iterations": 64,
        "default_games": 256,
        "default_simulations": 200,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 25,
        "c_puct": 2.5,
        "dirichlet_alpha": 1.4,   # ~10 / avg_legal_moves
        "dirichlet_epsilon": 0.4,
        "contempt_n": 20,
        "tree_reuse": True,
        "play_simulations": 200,
        "play_c_puct": 2.5,
    },

    "santorini": {
        # Network
        "num_filters": 128,
        "num_res_blocks": 5,
        "value_head_channels": 32,
        "value_head_fc_size": 64,
        "policy_head_channels": 4,
        "resblock_dropout": 0.1,

        # Training
        "train_ratio": 8,
        "buffer_size": 150000,
        "value_loss_weight": 1.5,
        "value_label_smoothing": 0.1,
        "surprise_kl_frac": 0.5,
        "lr": 0.005,
        "batch_size": 256,

        # Self-play
        "default_iterations": 64,
        "default_games": 256,
        "default_simulations": 256,
        "selects_per_round": 8,
        "vl_value": 0.0,
        "temp_threshold": 15,
        "c_puct": 2.5,
        "dirichlet_alpha": 0.25,  # ~10 / avg_legal_moves
        "dirichlet_epsilon": 0.4,
        "random_opening_moves": 4,
        "random_opening_fraction": 0.5,
        "contempt_n": 10,
        "tree_reuse": True,
        "play_simulations": 500,
        "play_c_puct": 1.5,
    },
}
