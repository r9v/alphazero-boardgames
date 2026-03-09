try:
    from .c_mcts import CMCTS as MCTS
except ImportError:
    from .mcts import MCTS
