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

        # --- Self-play ---
        "default_iterations": 24,
        "default_games": 24,
        "default_simulations": 24,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 3,
        "c_puct": 1.5,

        # --- Play (human vs AI) ---
        "play_simulations": 16,
        "play_c_puct": 1.5,
    },

    "connect4": {
        # --- Network ---
        "num_filters": 64,
        "num_res_blocks": 2,
        "value_head_channels": 2,
        "value_head_fc_size": 32,
        "policy_head_channels": 2,

        # --- Training ---
        "max_train_steps": 3_200,
        "target_epochs": 4,
        "buffer_size": 12_000,
        "value_loss_weight": 3.0,

        # --- Self-play ---
        "default_iterations": 32,
        "default_games": 64,
        "default_simulations": 64,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 15,
        "c_puct": 1.5,

        # --- Play (human vs AI) ---
        "play_simulations": 400,
        "play_c_puct": 2.5,
    },

    "santorini": {
        # --- Network ---
        "num_filters": 128,
        "num_res_blocks": 2,
        "value_head_channels": 4,
        "value_head_fc_size": 64,
        "policy_head_channels": 4,

        # --- Training ---
        "max_train_steps": 12800,
        "target_epochs": 4,
        "buffer_size": 55000,
        "value_loss_weight": 3.0,

        # --- Self-play ---
        "default_iterations": 64,
        "default_games": 64,
        "default_simulations": 256,
        "selects_per_round": 8,
        "vl_value": 3.0,
        "temp_threshold": 15,
        "c_puct": 1.5,

        # --- Play (human vs AI) ---
        "play_simulations": 400,
        "play_c_puct": 2.0,
    },
}
