from .alphazero_net import AlphaZeroNet

GAME_CONFIGS = {
    "tictactoe": {
        "num_filters": 64, "num_res_blocks": 2,
        "max_train_steps": 3200,
        "target_epochs": 4, "buffer_size": 50000,
        "default_iterations": 10, "default_games": 32, "default_simulations": 32,
        "selects_per_round": 1, "vl_value": 0.0,
        "value_loss_weight": 1.0,
        "temp_threshold": 4, "c_puct": 1.5,
    },
    "connect4": {
        "num_filters": 128, "num_res_blocks": 3,
        "max_train_steps": 6400,
        "target_epochs": 4, "buffer_size": 25000,
        "default_iterations": 32, "default_games": 64, "default_simulations": 200,
        "selects_per_round": 1, "vl_value": 0.0,
        "value_loss_weight": 3.0,
        "temp_threshold": 15, "c_puct": 1.5,
    },
    "santorini": {
        "num_filters": 256, "num_res_blocks": 3,
        "max_train_steps": 12800,
        "target_epochs": 4, "buffer_size": 200000,
        "default_iterations": 10, "default_games": 64, "default_simulations": 128,
        "selects_per_round": 8, "vl_value": 3.0,
        "value_loss_weight": 5.0,
        "temp_threshold": 15, "c_puct": 1.5,
    },
}
