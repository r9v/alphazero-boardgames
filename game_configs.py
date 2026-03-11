GAME_CONFIGS = {
    "tictactoe": {
        # --- Network ---
        "num_filters": 64,
        "num_res_blocks": 2,

        # --- Training ---
        "max_train_steps": 3200,
        "target_epochs": 4,
        "buffer_size": 10000,
        "value_loss_weight": 1.0,

        # --- Self-play ---
        "default_iterations": 10,
        "default_games": 32,
        "default_simulations": 32,
        "selects_per_round": 1,
        "vl_value": 0.0,
        "temp_threshold": 5,
        "c_puct": 1.5,

        # --- Play (human vs AI) ---
        "play_simulations": 100,
        "play_c_puct": 1.5,
    },

    "connect4": {
        # --- Network ---
        "num_filters": 128,
        "num_res_blocks": 3,

        # --- Training ---
        "max_train_steps": 6400,
        "target_epochs": 4,
        "buffer_size": 25000,
        "value_loss_weight": 3.0,

        # --- Self-play ---
        "default_iterations": 32,
        "default_games": 64,
        "default_simulations": 128,
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
        "num_filters": 256,
        "num_res_blocks": 3,
        "value_head_channels": 8,
        "value_head_fc_size": 128,

        # --- Training ---
        "max_train_steps": 12800,
        "target_epochs": 4,
        "buffer_size": 55000,
        "value_loss_weight": 1.0, 

        # --- Self-play ---
        "default_iterations": 32,
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
