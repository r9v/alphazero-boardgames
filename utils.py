"""Shared utilities used across train.py, play.py, and self-play."""
import importlib

import torch.nn.functional as F

from game_configs import GAME_CONFIGS


def wdl_to_scalar(wdl_logits):
    """Convert WDL logits [B, 3] to scalar values [B]: P(win) - P(loss).

    Works with both torch tensors and numpy arrays.
    """
    probs = F.softmax(wdl_logits, dim=1) if hasattr(wdl_logits, 'dim') else wdl_logits
    return probs[:, 0] - probs[:, 2]


GAMES = {
    "tictactoe": "games.tictactoe:TTTGame",
    "connect4": "games.connect4:Connect4Game",
    "santorini": "games.santorini:SantoriniGame",
}


def load_game(name):
    """Import and instantiate a game by name."""
    if name not in GAMES:
        raise ValueError(f"Unknown game '{name}'. Choose from: {list(GAMES.keys())}")
    module_path, class_name = GAMES[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)()


def make_net(game, game_name):
    """Build an AlphaZeroNet from game config defaults."""
    from network import AlphaZeroNet
    cfg = GAME_CONFIGS.get(game_name, {})
    input_channels = getattr(game, 'input_channels',
                             2 * (game.num_history_states + 1))
    return AlphaZeroNet(
        input_channels=input_channels,
        board_shape=game.board_shape,
        action_size=game.action_size,
        num_res_blocks=cfg.get("num_res_blocks", 2),
        num_filters=cfg.get("num_filters", 256),
        value_head_channels=cfg.get("value_head_channels", 2),
        value_head_fc_size=cfg.get("value_head_fc_size", 64),
        policy_head_channels=cfg.get("policy_head_channels", 2),
        backbone_dropout=cfg.get("backbone_dropout", 0.15),
        num_groups=cfg.get("num_groups", 8),
    )


def log_backends(mcts_cls, game):
    """Print MCTS and game backend info. Returns (mcts_label, game_label)."""
    mcts_mod = mcts_cls.__module__
    mcts_label = "C/Cython" if "c_mcts" in mcts_mod else "Python"
    game_mod = type(game).__module__
    game_label = "C/Cython" if "c_game" in game_mod else "Python"
    print(f"  MCTS backend: {mcts_label} ({mcts_mod})")
    print(f"  Game backend: {game_label} ({game_mod})")
    return mcts_label, game_label
