"""Tests for Santorini worker placement phase."""
import pytest
import numpy as np
from games.santorini import SantoriniGame, BOARD_SIZE


@pytest.fixture
def game():
    return SantoriniGame()


# --- 1. New game starts in placement phase ---

class TestNewGame:
    def test_placed_count_zero(self, game):
        s = game.new_game()
        assert s.placed_count == 0

    def test_player_is_p1(self, game):
        s = game.new_game()
        assert s.player == -1

    def test_25_available_actions(self, game):
        s = game.new_game()
        assert s.available_actions.sum() == 25

    def test_actions_are_0_to_24(self, game):
        s = game.new_game()
        for a in range(25):
            assert s.available_actions[a] == 1
        for a in range(25, 128):
            assert s.available_actions[a] == 0

    def test_empty_workers(self, game):
        s = game.new_game()
        assert s.workers[-1] == []
        assert s.workers[1] == []

    def test_not_terminal(self, game):
        s = game.new_game()
        assert not s.terminal

    def test_empty_board(self, game):
        s = game.new_game()
        assert (s.board == 0).all()


# --- 2. Placement turn sequence ---

class TestPlacementSequence:
    def _place_sequence(self, game, positions):
        """Place workers at given positions, return list of states."""
        s = game.new_game()
        states = [s]
        for r, c in positions:
            s = game.step(s, r * 5 + c)
            states.append(s)
        return states

    def test_full_sequence_players(self, game):
        """P1, P1, P2, P2, then P1 starts normal play."""
        states = self._place_sequence(game,
                                      [(0, 0), (0, 4), (4, 0), (4, 4)])
        # placed_count 0->1: P1 stays
        assert states[1].player == -1
        # placed_count 1->2: switch to P2
        assert states[2].player == 1
        # placed_count 2->3: P2 stays
        assert states[3].player == 1
        # placed_count 3->4: switch to P1
        assert states[4].player == -1

    def test_full_sequence_placed_count(self, game):
        states = self._place_sequence(game,
                                      [(0, 0), (0, 4), (4, 0), (4, 4)])
        for i, s in enumerate(states):
            assert s.placed_count == i

    def test_last_turn_skipped(self, game):
        states = self._place_sequence(game,
                                      [(0, 0), (0, 4), (4, 0), (4, 4)])
        assert states[0].last_turn_skipped is False  # game start
        assert states[1].last_turn_skipped is True   # P1 again
        assert states[2].last_turn_skipped is False   # switch to P2
        assert states[3].last_turn_skipped is True   # P2 again
        assert states[4].last_turn_skipped is False   # switch to P1

    def test_workers_accumulate(self, game):
        states = self._place_sequence(game,
                                      [(2, 2), (0, 0), (4, 4), (4, 0)])
        assert states[1].workers[-1] == [(2, 2)]
        assert states[1].workers[1] == []

        assert states[2].workers[-1] == [(2, 2), (0, 0)]
        assert states[2].workers[1] == []

        assert states[3].workers[-1] == [(2, 2), (0, 0)]
        assert states[3].workers[1] == [(4, 4)]

        assert states[4].workers[-1] == [(2, 2), (0, 0)]
        assert states[4].workers[1] == [(4, 4), (4, 0)]

    def test_available_actions_decrease(self, game):
        states = self._place_sequence(game,
                                      [(2, 2), (0, 0), (4, 4), (4, 0)])
        assert states[0].available_actions.sum() == 25
        assert states[1].available_actions.sum() == 24
        assert states[2].available_actions.sum() == 23
        assert states[3].available_actions.sum() == 22

    def test_occupied_cell_not_available(self, game):
        s = game.new_game()
        s = game.step(s, 2 * 5 + 2)  # place at (2,2)
        assert s.available_actions[2 * 5 + 2] == 0  # can't place there again


# --- 3. Transition to normal gameplay ---

class TestPlacementToNormal:
    def test_normal_actions_after_placement(self, game):
        s = game.new_game()
        for r, c in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            s = game.step(s, r * 5 + c)

        assert s.placed_count == 4
        assert not s.terminal
        # Normal actions should be in range 0-127 with move/build encoding
        assert s.available_actions.sum() > 0
        # No placement actions should dominate - check that actions > 24 exist
        assert s.available_actions[25:].sum() > 0

    def test_can_play_normal_move(self, game):
        s = game.new_game()
        for r, c in [(0, 0), (0, 4), (4, 0), (4, 4)]:
            s = game.step(s, r * 5 + c)

        # Find a valid action and play it
        valid = np.where(s.available_actions == 1)[0]
        assert len(valid) > 0
        s2 = game.step(s, int(valid[0]))
        assert s2.placed_count == 4  # stays at 4
        assert s2.player == 1  # switches to P2

    def test_state_to_input_during_placement(self, game):
        """state_to_input should handle 0-1 workers without error."""
        s = game.new_game()
        inp = game.state_to_input(s)
        assert inp.shape == (7, 5, 5)
        # All building levels should be 0
        assert inp[0].sum() == 25  # level 0 everywhere
        # No workers
        assert inp[5].sum() == 0
        assert inp[6].sum() == 0

    def test_state_to_input_partial_placement(self, game):
        s = game.new_game()
        s = game.step(s, 2 * 5 + 3)  # P1 places at (2,3)
        inp = game.state_to_input(s)
        # P1 placed one worker, still P1's turn (skip)
        # Channel 5 = current player's workers
        assert inp[5].sum() == 1
        assert inp[5][2][3] == 1.0
        assert inp[6].sum() == 0


# --- 4. Invalid placement actions ---

class TestPlacementValidation:
    def test_invalid_action_raises(self, game):
        s = game.new_game()
        with pytest.raises(ValueError):
            game.step(s, 25)  # action 25 not available during placement

    def test_occupied_cell_raises(self, game):
        s = game.new_game()
        s = game.step(s, 0)  # place at (0,0)
        with pytest.raises(ValueError):
            game.step(s, 0)  # can't place at (0,0) again


# --- 5. Symmetry during placement ---

class TestPlacementSymmetry:
    def test_symmetry_count(self, game):
        """get_symmetries returns 8 results during placement."""
        s = game.new_game()
        inp = game.state_to_input(s)
        policy = np.zeros(128, dtype=np.float32)
        policy[12] = 1.0  # place at (2,2)

        syms = game.get_symmetries(inp, policy)
        assert len(syms) == 8

    def test_placement_policy_remapped(self, game):
        """Placement action (2,2) center should stay at center for all syms."""
        s = game.new_game()
        inp = game.state_to_input(s)
        policy = np.zeros(128, dtype=np.float32)
        policy[2 * 5 + 2] = 1.0  # center cell

        syms = game.get_symmetries(inp, policy)
        for new_state, new_policy in syms:
            # Center (2,2) maps to (2,2) under all D4 transforms
            assert new_policy[2 * 5 + 2] == 1.0
            assert new_policy.sum() == pytest.approx(1.0)

    def test_placement_corner_transforms(self, game):
        """Corner (0,0) should map to all 4 corners under rotations."""
        s = game.new_game()
        inp = game.state_to_input(s)
        policy = np.zeros(128, dtype=np.float32)
        policy[0 * 5 + 0] = 1.0  # (0,0)

        syms = game.get_symmetries(inp, policy)
        # Collect where the mass ends up
        targets = set()
        for _, new_policy in syms:
            idx = np.argmax(new_policy)
            targets.add(idx)
            assert new_policy.sum() == pytest.approx(1.0)

        # (0,0), (0,4), (4,0), (4,4) = actions 0, 4, 20, 24
        assert {0, 4, 20, 24}.issubset(targets)

    def test_placement_mass_preserved(self, game):
        """Total policy mass is preserved during placement symmetry."""
        s = game.new_game()
        inp = game.state_to_input(s)
        policy = np.random.dirichlet(np.ones(25)).astype(np.float32)
        full_policy = np.zeros(128, dtype=np.float32)
        full_policy[:25] = policy

        syms = game.get_symmetries(inp, full_policy)
        for _, new_policy in syms:
            assert new_policy.sum() == pytest.approx(full_policy.sum(), abs=1e-5)

    def test_partial_placement_symmetry(self, game):
        """Symmetry works with 1 worker placed."""
        s = game.new_game()
        s = game.step(s, 0)  # P1 places at (0,0)
        inp = game.state_to_input(s)
        policy = np.zeros(128, dtype=np.float32)
        # Uniform over remaining 24 cells
        for a in range(25):
            if s.available_actions[a]:
                policy[a] = 1.0 / 24

        syms = game.get_symmetries(inp, policy)
        assert len(syms) == 8
        for _, new_policy in syms:
            assert new_policy.sum() == pytest.approx(1.0, abs=1e-5)


# --- 6. Full game with placement ---

class TestFullGameWithPlacement:
    def test_play_random_game(self, game):
        """Play a full random game starting from placement."""
        s = game.new_game()
        moves = 0
        while not s.terminal:
            valid = np.where(s.available_actions == 1)[0]
            assert len(valid) > 0
            action = np.random.choice(valid)
            s = game.step(s, action)
            moves += 1
            assert moves < 200  # sanity: game should end

        assert s.terminal
        assert s.terminal_value in (-1, 1)
        assert moves >= 5  # at least 4 placements + 1 move

    def test_play_10_random_games(self, game):
        """Play 10 random games to stress-test placement + gameplay."""
        for _ in range(10):
            s = game.new_game()
            moves = 0
            while not s.terminal:
                valid = np.where(s.available_actions == 1)[0]
                action = np.random.choice(valid)
                s = game.step(s, action)
                moves += 1
                if moves > 200:
                    break
            assert s.terminal


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
