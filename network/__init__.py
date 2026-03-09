from .alphazero_net import AlphaZeroNet

GAME_CONFIGS = {
    "tictactoe": {
        "num_filters": 64, "num_res_blocks": 2,
        "max_train_steps": 500,
    },
    "connect4": {
        "num_filters": 128, "num_res_blocks": 3,
        "max_train_steps": 1000,
    },
    "santorini": {
        "num_filters": 128, "num_res_blocks": 3,
        "max_train_steps": 2000,
    },
}
