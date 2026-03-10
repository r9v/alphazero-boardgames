from .alphazero_net import AlphaZeroNet

GAME_CONFIGS = {
    "tictactoe": {
        "num_filters": 64, "num_res_blocks": 2,
        "max_train_steps": 2000,
        "target_epochs": 4, "buffer_size": 50000,
        "selects_per_round": 1, "vl_value": 0.0,
        "value_loss_weight": 1.0,
        "temp_threshold": 4, "c_puct": 1.5,
    },
    "connect4": {
        "num_filters": 128, "num_res_blocks": 3,
        "max_train_steps": 5000,
        "target_epochs": 4, "buffer_size": 100000,
        "selects_per_round": 1, "vl_value": 0.0,
        "value_loss_weight": 2.0,
        "temp_threshold": 10, "c_puct": 1.5,
    },
    "santorini": {
        "num_filters": 256, "num_res_blocks": 3,
        "max_train_steps": 8000,
        "target_epochs": 4, "buffer_size": 200000,
        "selects_per_round": 8, "vl_value": 3.0,
        "value_loss_weight": 4.0,
        "temp_threshold": 15, "c_puct": 1.5,
    },
}
