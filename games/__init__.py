from .tictactoe import TTTGame

try:
    from .c_connect4 import CConnect4Game as Connect4Game
except ImportError:
    from .connect4 import Connect4Game
