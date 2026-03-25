import argparse
import pygame
import sys
import threading
import numpy as np
from . import SantoriniGame, DIRECTIONS, BOARD_SIZE

game = SantoriniGame()

# --- Layout ---
CELL = 100
PAD = 50
TOP_BAR = 50
BOARD_PX = CELL * BOARD_SIZE
WIN_W = BOARD_PX + PAD * 2
WIN_H = BOARD_PX + PAD * 2 + TOP_BAR

# --- Colors ---
BG = (245, 245, 245)
GRID_LINE = (180, 180, 180)
GROUND = (228, 228, 228)

# Building level colors (progressively warmer)
LEVEL_COLORS = [
    (228, 228, 228),  # 0 - ground gray
    (253, 230, 138),  # 1 - light yellow
    (251, 191, 36),   # 2 - amber
    (146, 64, 14),    # 3 - brown
]
DOME_COLOR = (101, 67, 33)  # deep brown

# Workers
P1_FILL = (255, 251, 235)    # warm white
P1_OUTLINE = (217, 119, 6)   # amber
P2_FILL = (55, 65, 81)       # dark charcoal
P2_OUTLINE = (17, 24, 39)    # near-black

# Highlights
HIGHLIGHT = (187, 247, 208, 150)      # green (semi-transparent)
SELECTED_HIGHLIGHT = (249, 115, 22)   # orange

# Text
TEXT_COLOR = (30, 30, 30)
WIN_COLOR = (220, 38, 38)


def grid_to_pixel(r, c):
    """Convert grid (row, col) to pixel center of cell."""
    x = PAD + c * CELL + CELL // 2
    y = TOP_BAR + PAD + r * CELL + CELL // 2
    return x, y


def pixel_to_grid(mx, my):
    """Convert mouse pixel to grid (row, col) or None."""
    c = (mx - PAD) // CELL
    r = (my - TOP_BAR - PAD) // CELL
    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
        return int(r), int(c)
    return None


def draw_board(screen, state, phase, selected_worker_pos, valid_targets,
               ai_thinking=False):
    """Draw the full board."""
    screen.fill(BG)
    font = pygame.font.SysFont("Arial", 18, bold=True)
    small_font = pygame.font.SysFont("Arial", 13)

    # Status bar
    if state.terminal:
        if state.terminal_value == 0:
            status = "Draw!"
        else:
            winner = "1 (White)" if state.terminal_value == -1 else "2 (Black)"
            status = f"Player {winner} wins!"
        color = WIN_COLOR
    elif hasattr(state, 'placed_count') and state.placed_count < 4:
        name = "1 (White)" if state.player == -1 else "2 (Black)"
        worker_num = len(state.workers[state.player]) + 1
        if ai_thinking:
            status = f"Player {name}  —  AI placing worker..."
        else:
            status = f"Player {name}  —  place worker {worker_num}"
        color = TEXT_COLOR
    else:
        name = "1 (White)" if state.player == -1 else "2 (Black)"
        if ai_thinking:
            status = f"Player {name}  —  AI thinking..."
        else:
            status = f"Player {name}'s turn"
            if phase == "WORKER_SELECTED":
                status += "  —  select move"
            elif phase == "MOVE_SELECTED":
                status += "  —  select build"
        color = TEXT_COLOR
    text_surf = font.render(status, True, color)
    screen.blit(text_surf, (PAD, 15))

    # Draw cells
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x, y = grid_to_pixel(r, c)
            rect = pygame.Rect(x - CELL // 2, y - CELL // 2, CELL, CELL)

            # Ground
            pygame.draw.rect(screen, GROUND, rect)
            pygame.draw.rect(screen, GRID_LINE, rect, 1)

            level = state.board[r][c]

            # Draw building layers
            if level >= 4:
                # Dome: draw levels 1-3 then dome on top
                for lv in range(1, 4):
                    shrink = lv * 8
                    lrect = pygame.Rect(
                        x - CELL // 2 + shrink,
                        y - CELL // 2 + shrink,
                        CELL - shrink * 2,
                        CELL - shrink * 2,
                    )
                    pygame.draw.rect(screen, LEVEL_COLORS[lv], lrect)
                    pygame.draw.rect(screen, GRID_LINE, lrect, 1)
                # Dome triangle
                tri_points = [
                    (x, y - 16),       # top
                    (x - 14, y + 10),  # bottom-left
                    (x + 14, y + 10),  # bottom-right
                ]
                pygame.draw.polygon(screen, DOME_COLOR, tri_points)
                pygame.draw.polygon(screen, (70, 45, 20), tri_points, 2)
            elif level > 0:
                for lv in range(1, level + 1):
                    shrink = lv * 8
                    lrect = pygame.Rect(
                        x - CELL // 2 + shrink,
                        y - CELL // 2 + shrink,
                        CELL - shrink * 2,
                        CELL - shrink * 2,
                    )
                    pygame.draw.rect(screen, LEVEL_COLORS[lv], lrect)
                    pygame.draw.rect(screen, GRID_LINE, lrect, 1)

                # Level number
                lbl = small_font.render(str(level), True, (100, 100, 100))
                screen.blit(lbl, (x + CELL // 2 - 18, y - CELL // 2 + 4))

    # Draw highlights
    if valid_targets:
        highlight_surf = pygame.Surface((CELL - 4, CELL - 4), pygame.SRCALPHA)
        highlight_surf.fill(HIGHLIGHT)
        for (tr, tc) in valid_targets:
            tx, ty = grid_to_pixel(tr, tc)
            screen.blit(highlight_surf, (tx - CELL // 2 + 2, ty - CELL // 2 + 2))

    # Draw selected worker highlight
    if selected_worker_pos:
        sr, sc = selected_worker_pos
        sx, sy = grid_to_pixel(sr, sc)
        sel_rect = pygame.Rect(sx - CELL // 2 + 1, sy - CELL // 2 + 1,
                               CELL - 2, CELL - 2)
        pygame.draw.rect(screen, SELECTED_HIGHLIGHT, sel_rect, 3)

    # Draw workers
    for player in (-1, 1):
        fill = P1_FILL if player == -1 else P2_FILL
        outline = P1_OUTLINE if player == -1 else P2_OUTLINE
        for (wr, wc) in state.workers[player]:
            wx, wy = grid_to_pixel(wr, wc)
            pygame.draw.circle(screen, fill, (wx, wy), 18)
            pygame.draw.circle(screen, outline, (wx, wy), 18, 3)

    pygame.display.flip()


class GUI:
    def __init__(self, ai_player=None, simulations=100, filters=256,
                 res_blocks=2, c_puct=1.5,
                 value_head_channels=2, value_head_fc_size=64,
                 policy_head_channels=2):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Santorini")
        # Bring window to front
        import ctypes
        hwnd = pygame.display.get_wm_info()["window"]
        ctypes.windll.user32.SetForegroundWindow(hwnd)

        self.state = game.new_game()
        self.phase = "PLACEMENT"
        self.selected_worker_idx = None
        self.selected_worker_pos = None
        self.selected_move_dir = None
        self.selected_move_target = None
        self.valid_targets = {}  # {(r,c): [dir_indices...]}
        self.move_number = 0

        # AI setup
        self.ai_player = ai_player
        self.ai_thinking = False
        self.ai_thread = None
        self.ai_result = None
        self.simulations = simulations
        self.mcts = None

        if self.ai_player is not None:
            from network import AlphaZeroNet
            from mcts import MCTS

            net = AlphaZeroNet(
                input_channels=game.input_channels,
                board_shape=game.board_shape,
                action_size=game.action_size,
                num_res_blocks=res_blocks,
                num_filters=filters,
                value_head_channels=value_head_channels,
                value_head_fc_size=value_head_fc_size,
                policy_head_channels=policy_head_channels,
            )
            checkpoint_dir = "checkpoints/santorini"
            loaded_path = net.load_latest(checkpoint_dir)
            if loaded_path:
                print(f"Loaded model: {loaded_path}")
            else:
                print("No checkpoint found, using untrained network.")

            mcts_mod = MCTS.__module__
            mcts_label = "C/Cython" if "c_mcts" in mcts_mod else "Python"
            game_mod = type(game).__module__
            game_label = "C/Cython" if "c_game" in game_mod else "Python"
            human_side = "White (P1)" if ai_player == 1 else "Black (P2)"
            print(f"Config: sims={simulations} c_puct={c_puct}")
            print(f"  MCTS backend: {mcts_label} ({mcts_mod})")
            print(f"  Game backend: {game_label} ({game_mod})")
            print(f"  Human: {human_side}")
            self.mcts = MCTS(game, net, c_puct=c_puct)

        self._log_board()
        self._draw()
        self._run()

    def _log_board(self):
        """Print board state and worker positions to console."""
        s = self.state
        print(f"\n  Board (levels):")
        for r in range(BOARD_SIZE):
            row = " ".join(str(s.board[r][c]) for c in range(BOARD_SIZE))
            print(f"    {row}")
        for p, label in [(-1, "White(P1)"), (1, "Black(P2)")]:
            ws = s.workers[p]
            print(f"  {label} workers: {[(r,c) for r,c in ws]}")
        if not s.terminal:
            turn = "White(P1)" if s.player == -1 else "Black(P2)"
            print(f"  Turn: {turn}")

    def _draw(self):
        if self.phase == "PLACEMENT":
            # Highlight all empty cells as valid placement targets
            targets = set()
            for action in range(25):
                if self.state.available_actions[action]:
                    targets.add((action // 5, action % 5))
            draw_board(self.screen, self.state, self.phase,
                       None, targets, ai_thinking=self.ai_thinking)
        else:
            targets = set(self.valid_targets.keys()) if self.valid_targets else None
            draw_board(self.screen, self.state, self.phase,
                       self.selected_worker_pos, targets,
                       ai_thinking=self.ai_thinking)

    def _reset_phase(self):
        if hasattr(self.state, 'placed_count') and self.state.placed_count < 4:
            self.phase = "PLACEMENT"
        else:
            self.phase = "IDLE"
        self.selected_worker_idx = None
        self.selected_worker_pos = None
        self.selected_move_dir = None
        self.selected_move_target = None
        self.valid_targets = {}

    def _log_terminal(self):
        if self.state.terminal:
            if self.state.terminal_value == 0:
                print(f"\n=== GAME OVER: Draw ({self.move_number} moves) ===")
            else:
                winner = "White(P1)" if self.state.terminal_value == -1 else "Black(P2)"
                print(f"\n=== GAME OVER: {winner} wins ({self.move_number} moves) ===")

    def _advance_tree(self, action):
        """Advance MCTS tree to the child of the taken action for reuse."""
        if self.mcts is None or self.mcts.last_root is None:
            return
        children = self.mcts.last_root.children
        child = children[action]
        if child is not None and child.P is not None:
            child.parent = None
            self.mcts.last_root = child
        else:
            self.mcts.last_root = None

    def _is_ai_turn(self):
        return (self.ai_player is not None
                and not self.state.terminal
                and self.state.player == self.ai_player)

    def _start_ai_move(self):
        """Start AI computation in background thread."""
        import time

        state_snapshot = self.state
        is_placement = (hasattr(state_snapshot, 'placed_count')
                        and state_snapshot.placed_count < 4)

        reuse_root = self.mcts.last_root

        def compute():
            t0 = time.time()
            pi = self.mcts.get_policy(self.simulations, state_snapshot,
                                      root_node=reuse_root)
            elapsed = time.time() - t0
            move = int(np.argmax(pi))
            self.ai_result = (pi, move, elapsed, is_placement)

        self.ai_thread = threading.Thread(target=compute, daemon=True)
        self.ai_thread.start()

    def _finish_ai_move(self):
        """Apply AI move result to game state."""
        pi, move, elapsed, is_placement = self.ai_result
        self.ai_result = None
        self.ai_thread = None

        # Log AI move
        self.move_number += 1
        root = self.mcts.last_root
        player = "White(P1)" if self.state.player == -1 else "Black(P2)"

        if is_placement:
            r, c = move // 5, move % 5
            worker_num = len(self.state.workers[self.state.player]) + 1
            print(f"\n--- Move {self.move_number}: AI ({player}) places "
                  f"worker {worker_num} at ({r},{c}) "
                  f"| {self.simulations} sims in {elapsed:.2f}s ---")
            print(f"  Root N={root.n}  V={root.nnet_value:+.4f}")

            # Show top placement actions
            actions_with_visits = []
            children = root.children
            for a in range(25):
                child = children[a]
                if child is not None and child.n > 0:
                    actions_with_visits.append((a, child.n, child.Q))
            actions_with_visits.sort(key=lambda x: -x[1])
            for a, n, q in actions_with_visits[:10]:
                ar, ac = a // 5, a % 5
                marker = " <--" if a == move else ""
                print(f"    ({ar},{ac}) N={n:>5} Q={q:+.4f}{marker}")
        else:
            print(f"\n--- Move {self.move_number}: AI ({player}) action={move} "
                  f"| {self.simulations} sims in {elapsed:.2f}s ---")
            print(f"  Root N={root.n}  V={root.nnet_value:+.4f}")

            # Decode and show top actions by visit count
            actions_with_visits = []
            children = root.children
            for a in range(128):
                child = children[a]
                if child is not None and child.n > 0:
                    actions_with_visits.append((a, child.n, child.Q))

            actions_with_visits.sort(key=lambda x: -x[1])
            top = actions_with_visits[:10]

            print(f"  {'Action':>6}  {'N':>6}  {'Q':>8}  {'P':>7}  "
                  f"{'pi':>7}  Description")
            print(f"  {'------':>6}  {'---':>6}  {'---':>8}  {'---':>7}  "
                  f"{'---':>7}  -----------")
            for a, n, q in top:
                w_idx = a // 64
                m_dir = (a % 64) // 8
                b_dir = a % 8
                my_workers = self.state._sorted_workers(self.state.player)
                wr, wc = my_workers[w_idx]
                dr, dc = DIRECTIONS[m_dir]
                mr, mc = wr + dr, wc + dc
                bdr, bdc = DIRECTIONS[b_dir]
                br, bc = mr + bdr, mc + bdc
                desc = f"W({wr},{wc})→({mr},{mc}) B({br},{bc})"
                p_val = (f"{root.P[a]:.4f}" if root.P[a] > 0.0001
                         else "  .   ")
                pi_val = (f"{pi[a]:.4f}" if pi[a] > 0.001
                          else "  .   ")
                marker = " <--" if a == move else ""
                print(f"  {a:>6}  {n:>6}  {q:>+8.4f}  {p_val:>7}  "
                      f"{pi_val:>7}  {desc}{marker}")

            print(f"  Total unique actions visited: "
                  f"{len(actions_with_visits)}")

        self._advance_tree(move)
        self.state = game.step(self.state, move)
        self._log_board()
        self._log_terminal()
        self._reset_phase()
        self.ai_thinking = False

    def _on_click(self, pos):
        if self.state.terminal or self._is_ai_turn():
            return

        cell = pixel_to_grid(*pos)
        if cell is None:
            self._reset_phase()
            self._draw()
            return

        r, c = cell

        if self.phase == "PLACEMENT":
            self._handle_placement(r, c)
        elif self.phase == "IDLE":
            self._handle_idle(r, c)
        elif self.phase == "WORKER_SELECTED":
            self._handle_worker_selected(r, c)
        elif self.phase == "MOVE_SELECTED":
            self._handle_move_selected(r, c)

        self._draw()

    def _handle_placement(self, r, c):
        """Click an empty cell to place a worker."""
        action = r * 5 + c
        if self.state.available_actions[action] != 1:
            return

        player = "White(P1)" if self.state.player == -1 else "Black(P2)"
        worker_num = len(self.state.workers[self.state.player]) + 1
        print(f"\n--- Placement: Human ({player}) worker {worker_num} "
              f"at ({r},{c}) ---")

        self._advance_tree(action)
        self.state = game.step(self.state, action)
        self._log_board()
        self._reset_phase()

    def _handle_idle(self, r, c):
        """Click a worker to select it."""
        my_workers = self.state._sorted_workers(self.state.player)
        for w_idx, (wr, wc) in enumerate(my_workers):
            if (r, c) == (wr, wc):
                # Compute valid move targets for this worker
                move_targets = {}
                for m_dir in range(8):
                    for b_dir in range(8):
                        action = w_idx * 64 + m_dir * 8 + b_dir
                        if self.state.available_actions[action]:
                            dr, dc = DIRECTIONS[m_dir]
                            target = (wr + dr, wc + dc)
                            if target not in move_targets:
                                move_targets[target] = []
                            move_targets[target].append(m_dir)

                if not move_targets:
                    return  # This worker has no moves

                self.phase = "WORKER_SELECTED"
                self.selected_worker_idx = w_idx
                self.selected_worker_pos = (wr, wc)
                self.valid_targets = move_targets
                return

    def _handle_worker_selected(self, r, c):
        """Click a move target or another worker."""
        if (r, c) in self.valid_targets:
            # Valid move — compute build targets
            m_dirs = self.valid_targets[(r, c)]
            m_dir = m_dirs[0]  # all map to same destination

            build_targets = {}
            for b_dir in range(8):
                action = self.selected_worker_idx * 64 + m_dir * 8 + b_dir
                if self.state.available_actions[action]:
                    bdr, bdc = DIRECTIONS[b_dir]
                    bt = (r + bdr, c + bdc)
                    if bt not in build_targets:
                        build_targets[bt] = []
                    build_targets[bt].append(b_dir)

            self.phase = "MOVE_SELECTED"
            self.selected_move_dir = m_dir
            self.selected_move_target = (r, c)
            self.valid_targets = build_targets

            # Auto-complete if only one build option
            if len(build_targets) == 1:
                only_target = list(build_targets.keys())[0]
                self._do_build(*only_target)
        else:
            # Maybe clicked another worker
            self._reset_phase()
            self._handle_idle(r, c)

    def _handle_move_selected(self, r, c):
        """Click a build target."""
        if (r, c) in self.valid_targets:
            self._do_build(r, c)
        else:
            self._reset_phase()

    def _do_build(self, br, bc):
        b_dir = self.valid_targets[(br, bc)][0]
        action = self.selected_worker_idx * 64 + self.selected_move_dir * 8 + b_dir

        # Log human move
        self.move_number += 1
        wr, wc = self.selected_worker_pos
        mr, mc = self.selected_move_target
        player = "White(P1)" if self.state.player == -1 else "Black(P2)"
        print(f"\n--- Move {self.move_number}: Human ({player}) action={action} ---")
        print(f"  W({wr},{wc})→({mr},{mc}) Build({br},{bc})")

        self._advance_tree(action)
        self.state = game.step(self.state, action)
        self._log_board()
        self._log_terminal()
        self._reset_phase()

    def _run(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if not self.ai_thinking:
                        self._on_click(event.pos)

            # Start AI turn in background thread
            if self._is_ai_turn() and not self.ai_thinking:
                self.ai_thinking = True
                self._draw()
                self._start_ai_move()

            # Check if AI finished
            if self.ai_thread and not self.ai_thread.is_alive():
                self._finish_ai_move()
                self._draw()

            clock.tick(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Santorini GUI")
    parser.add_argument("--ai", action="store_true",
                        help="Play against AlphaZero AI")
    parser.add_argument("--human-first", action="store_true",
                        help="Human plays first (as White)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS simulations per AI move")
    parser.add_argument("--filters", type=int, default=256)
    parser.add_argument("--res-blocks", type=int, default=2)
    args = parser.parse_args()

    ai_player = None
    if args.ai:
        # Human first = human is -1 (White), AI is 1 (Black)
        ai_player = 1 if args.human_first else -1

    GUI(ai_player=ai_player, simulations=args.simulations,
        filters=args.filters, res_blocks=args.res_blocks)
