from .alphazero_net import AlphaZeroNet

GAME_CONFIGS = {
    "tictactoe": {
        "num_filters": 64, "num_res_blocks": 2,
        "max_train_steps": 500,
        "selects_per_round": 1, "vl_value": 0.0,
        "value_loss_weight": 1.0,
    },
    "connect4": {
        "num_filters": 128, "num_res_blocks": 3,
        "max_train_steps": 1000,
        "selects_per_round": 1, "vl_value": 0.0,
        "value_loss_weight": 2.0,
    },
    "santorini": {
        "num_filters": 256, "num_res_blocks": 3,
        "max_train_steps": 2000,
        "selects_per_round": 8, "vl_value": 3.0,
        "value_loss_weight": 4.0,
    },
}
