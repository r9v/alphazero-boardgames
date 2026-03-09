from .alphazero_net import AlphaZeroNet

NETWORK_CONFIGS = {
    "tictactoe": {"num_filters": 64, "num_res_blocks": 2},
    "connect4": {"num_filters": 128, "num_res_blocks": 3},
    "santorini": {"num_filters": 256, "num_res_blocks": 5},
}
