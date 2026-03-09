try:
    from .c_game import CSantoriniGame as SantoriniGame
except ImportError:
    from .game import SantoriniGame
