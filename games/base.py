from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class GameState(ABC):
    """Abstract base for all game states.

    Subclasses must set these attributes in __init__:
        board:             np.ndarray of the current board
        player:            -1 or 1
        available_actions: binary mask (1 = legal), length = Game.action_size
        terminal:          bool
        terminal_value:    -1, 0, 1 when terminal, else None
        prev_state:        previous GameState (or None)
        last_turn_skipped: True if previous player had no moves
    """

    board: np.ndarray
    player: int
    available_actions: np.ndarray
    terminal: bool
    terminal_value: Optional[int]
    prev_state: Optional["GameState"]
    last_turn_skipped: bool


class Game(ABC):
    """Abstract base for all games."""

    board_shape: tuple          # e.g. (3,3), (6,7)
    action_size: int            # length of the action mask
    num_history_states: int     # how many prev board states to encode

    @abstractmethod
    def new_game(self) -> GameState:
        """Return the initial game state."""

    @abstractmethod
    def step(self, state: GameState, action: int) -> GameState:
        """Apply action and return the next game state."""

    @abstractmethod
    def state_to_input(self, state: GameState) -> np.ndarray:
        """Encode a game state into a tensor for the neural network.

        Returns an array of shape (C, *board_shape) where C is the
        number of input channels (derived from num_history_states).
        """

    def get_symmetries(self, state_input: np.ndarray, policy: np.ndarray):
        """Return list of (state_input, policy) including original + symmetries.

        Override in subclasses for games with board symmetries (e.g. mirror).
        """
        return [(state_input, policy)]
