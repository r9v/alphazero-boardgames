import tkinter as tk
import math
from .game import HiveGame, stack_size_and_top_piece
from .pieces import *

hive = HiveGame()

# Piece display: constant -> (letter, color)
PIECE_DISPLAY = {
    Player1A: ("A", "#3B82F6"),   # Ant - blue
    Player1G: ("G", "#22C55E"),   # Grasshopper - green
    Player1S: ("S", "#A16207"),   # Spider - brown
    Player1B: ("B", "#7C3AED"),   # Beetle - purple
    Player1Q: ("Q", "#EAB308"),   # Queen - gold
    Player2A: ("A", "#60A5FA"),   # Ant - light blue
    Player2G: ("G", "#4ADE80"),   # Grasshopper - light green
    Player2S: ("S", "#CA8A04"),   # Spider - light brown
    Player2B: ("B", "#A78BFA"),   # Beetle - light purple
    Player2Q: ("Q", "#FACC15"),   # Queen - light gold
}

P1_PIECES = (Player1A, Player1G, Player1S, Player1B, Player1Q)

# Player 1 pieces are 11-15, Player 2 are 21-25
P1_FILL = "#FFFBEB"   # warm white
P2_FILL = "#374151"   # dark charcoal
EMPTY_FILL = "#F3F4F6" # light gray
HIGHLIGHT_FILL = "#BBF7D0"  # light green for valid targets
SELECTED_OUTLINE = "#F97316"  # orange for selected piece

# Button colors per piece type
BUTTON_COLORS = {
    "Ant": "#3B82F6",
    "Grasshopper": "#22C55E",
    "Spider": "#A16207",
    "Beetle": "#7C3AED",
    "Queen": "#EAB308",
}

HEX_SIZE = 30


class InfoDisplay:
    def __init__(self, frame):
        self.player = tk.StringVar()
        self.turn = tk.StringVar()
        self.terminal = tk.StringVar()
        self.terminal_value = tk.StringVar()
        info_frame = tk.Frame(frame, pady=5)
        tk.Label(info_frame, textvariable=self.player, font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(info_frame, textvariable=self.turn, font=("Arial", 11)).pack(side=tk.LEFT)
        tk.Label(info_frame, textvariable=self.terminal, font=("Arial", 11)).pack(side=tk.LEFT)
        tk.Label(info_frame, textvariable=self.terminal_value, font=("Arial", 11)).pack(side=tk.LEFT)
        info_frame.pack()

    def update(self, player, turn, terminal, terminal_value):
        p_name = "1 (White)" if player == -1 else "2 (Black)"
        self.player.set(f'Player {p_name}  ')
        self.turn.set(f'Turn {turn}  ')
        self.terminal.set(f'Terminal: {terminal}  ' if terminal else '')
        self.terminal_value.set(f'Result: {terminal_value}' if terminal else '')


class HandDisplay:
    def __init__(self, frame, on_button_click):
        hand_frame = tk.Frame(frame, pady=5)

        left_frame = tk.Frame(hand_frame)
        right_frame = tk.Frame(hand_frame)
        left_frame.pack(side=tk.LEFT, padx=20)
        right_frame.pack(side=tk.RIGHT, padx=20)

        self.p1_label = tk.Label(left_frame, text="Player 1", font=("Arial", 10, "bold"))
        self.p1_label.pack(side=tk.TOP, pady=(0, 3))
        self.p2_label = tk.Label(right_frame, text="Player 2", font=("Arial", 10, "bold"))
        self.p2_label.pack(side=tk.TOP, pady=(0, 3))

        btn_opts = dict(width=12, font=("Arial", 9, "bold"), relief=tk.RAISED, bd=2)

        self.p1_buttons = {
            "Ant": tk.Button(left_frame, **btn_opts, command=lambda: on_button_click(Player1A)),
            "Grasshopper": tk.Button(left_frame, **btn_opts, command=lambda: on_button_click(Player1G)),
            "Spider": tk.Button(left_frame, **btn_opts, command=lambda: on_button_click(Player1S)),
            "Beetle": tk.Button(left_frame, **btn_opts, command=lambda: on_button_click(Player1B)),
            "Queen": tk.Button(left_frame, **btn_opts, command=lambda: on_button_click(Player1Q)),
        }

        self.p2_buttons = {
            "Ant": tk.Button(right_frame, **btn_opts, command=lambda: on_button_click(Player2A)),
            "Grasshopper": tk.Button(right_frame, **btn_opts, command=lambda: on_button_click(Player2G)),
            "Spider": tk.Button(right_frame, **btn_opts, command=lambda: on_button_click(Player2S)),
            "Beetle": tk.Button(right_frame, **btn_opts, command=lambda: on_button_click(Player2B)),
            "Queen": tk.Button(right_frame, **btn_opts, command=lambda: on_button_click(Player2Q)),
        }

        for name, btn in self.p1_buttons.items():
            btn.config(fg=BUTTON_COLORS[name])
            btn.pack(side=tk.TOP, pady=1)
        for name, btn in self.p2_buttons.items():
            btn.config(fg=BUTTON_COLORS[name])
            btn.pack(side=tk.TOP, pady=1)

        hand_frame.pack()

    def update_hand_display(self, player1_hand, player2_hand, active_player=1):
        counts_p1 = {
            "Ant": player1_hand.a, "Grasshopper": player1_hand.g,
            "Spider": player1_hand.s, "Beetle": player1_hand.b, "Queen": player1_hand.q,
        }
        counts_p2 = {
            "Ant": player2_hand.a, "Grasshopper": player2_hand.g,
            "Spider": player2_hand.s, "Beetle": player2_hand.b, "Queen": player2_hand.q,
        }

        for name, btn in self.p1_buttons.items():
            btn['text'] = f'{name} x{counts_p1[name]}'
        for name, btn in self.p2_buttons.items():
            btn['text'] = f'{name} x{counts_p2[name]}'

        # Highlight active player, disable inactive player's buttons
        if active_player == -1:
            self.p1_label.config(fg="#DC2626")
            self.p2_label.config(fg="black")
            for btn in self.p1_buttons.values():
                btn.config(state=tk.NORMAL)
            for btn in self.p2_buttons.values():
                btn.config(state=tk.DISABLED)
        else:
            self.p1_label.config(fg="black")
            self.p2_label.config(fg="#DC2626")
            for btn in self.p1_buttons.values():
                btn.config(state=tk.DISABLED)
            for btn in self.p2_buttons.values():
                btn.config(state=tk.NORMAL)


class Hex:
    def __init__(self, hex_id, x, y, canvas, canvas_x, canvas_y, dynamic=False):
        self.id = hex_id
        self.x = x
        self.y = y
        self.canvas = canvas
        self.canvas_x = canvas_x
        self.canvas_y = canvas_y
        self.text = None
        self.stack_text = None
        self.piece = None
        self.dynamic = dynamic

    def highlight(self):
        self.canvas.itemconfig(self.id, fill=HIGHLIGHT_FILL, outline='#16A34A', width=2)
        self.canvas.tag_raise(self.id)
        if self.text:
            self.canvas.tag_raise(self.text)
        if self.stack_text:
            self.canvas.tag_raise(self.stack_text)

    def clear_highlight(self):
        if self.piece:
            fill = P1_FILL if self.piece in P1_PIECES else P2_FILL
            self.canvas.itemconfig(self.id, fill=fill, outline='#9CA3AF', width=1)
        else:
            self.canvas.itemconfig(self.id, fill=EMPTY_FILL, outline='#D1D5DB', width=1)

    def remove_from_canvas(self):
        self.canvas.delete(self.id)
        if self.text:
            self.canvas.delete(self.text)
        if self.stack_text:
            self.canvas.delete(self.stack_text)

    def set_piece(self, piece, stack_size):
        self.piece = piece
        fill = P1_FILL if piece in P1_PIECES else P2_FILL
        self.canvas.itemconfig(self.id, fill=fill, outline='#9CA3AF', width=1)

        letter, color = PIECE_DISPLAY.get(piece, (str(piece), "black"))
        self.text = self.canvas.create_text(
            self.canvas_x, self.canvas_y,
            text=letter, fill=color, font=("Arial", 14, "bold"),
        )

        if stack_size > 1:
            self.stack_text = self.canvas.create_text(
                self.canvas_x + 10, self.canvas_y - 10,
                text=str(stack_size), fill="#EF4444", font=("Arial", 9, "bold"),
            )


class Hexes:
    def __init__(self):
        self.hexes = []
        self.highlighted = []

    def add(self, hex_obj):
        self.hexes.append(hex_obj)

    def get_by_id(self, target_id):
        return next((h for h in self.hexes if h.id == target_id or h.text == target_id), None)

    def get_by_xy(self, x, y):
        return next((h for h in self.hexes if h.x == x and h.y == y), None)

    def highlight(self, hex_obj):
        hex_obj.highlight()
        self.highlighted.append(hex_obj)

    def clear_highlighted(self):
        for h in self.highlighted:
            if h.dynamic:
                h.remove_from_canvas()
                self.hexes.remove(h)
            else:
                h.clear_highlight()
        self.highlighted = []


class GUI:
    def __init__(self):
        self.hexes = Hexes()
        self.piece_place_mode = False
        self.piece_to_place = None
        self.move_mode = False
        self.hex_to_move = None

        window = self._setup_window()

        self.state = hive.new_game()
        self.hand_display.update_hand_display(
            self.state.player1_hand, self.state.player2_hand, self.state.player)
        self.info_display.update(
            self.state.player, self.state.turn, self.state.terminal, self.state.terminal_value)
        self._draw_board()

        window.mainloop()

    def _grid_to_canvas(self, i, j):
        w = math.sqrt(3) * HEX_SIZE
        h = 2 * HEX_SIZE
        # Center grid (12, 12) at canvas center
        cx = 300 + (i - 12) * w + (j - 12) * w / 2
        cy = 275 + (j - 12) * 3 * h / 4
        return cx, cy

    def _draw_board(self):
        for j in range(25):
            for i in range(25):
                ss, piece = stack_size_and_top_piece(i, j, self.state.board)
                if ss != 0:
                    cx, cy = self._grid_to_canvas(i, j)
                    hex_id = self._draw_hex(cx, cy, HEX_SIZE)
                    hex_obj = Hex(hex_id, i, j, self.canvas, cx, cy)
                    hex_obj.set_piece(piece, ss)
                    self.hexes.add(hex_obj)

    def _draw_hex(self, x, y, size):
        points = []
        for i in range(6):
            angle = math.pi / 180 * (60 * i - 30)
            points.append(x + size * math.cos(angle))
            points.append(y + size * math.sin(angle))
        return self.canvas.create_polygon(
            points, width=1, fill=EMPTY_FILL, outline='#D1D5DB',
        )

    def _ensure_hex(self, i, j):
        existing = self.hexes.get_by_xy(i, j)
        if existing:
            return existing
        cx, cy = self._grid_to_canvas(i, j)
        hex_id = self._draw_hex(cx, cy, HEX_SIZE)
        hex_obj = Hex(hex_id, i, j, self.canvas, cx, cy, dynamic=True)
        self.hexes.add(hex_obj)
        return hex_obj

    def _on_canvas_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        hex_id = event.widget.find_overlapping(x, y, x + 1, y + 1)
        if len(hex_id) == 0:
            return
        self._on_hex_click(hex_id[0])

    def _on_hex_click(self, hex_id):
        hex_obj = self.hexes.get_by_id(hex_id)
        if hex_obj is None:
            return
        if self.piece_place_mode:
            self._try_place_piece(hex_obj)
        elif self.move_mode:
            self._try_move_piece(hex_obj)
        elif hex_obj.piece:
            self._start_move_action(hex_obj)

    def _start_move_action(self, hex_obj):
        move_actions = self.state.available_actions.get_move_actions_by_start(hex_obj.x, hex_obj.y)
        if len(move_actions) == 0:
            return
        self.move_mode = True
        self.hex_to_move = hex_obj
        self.canvas.itemconfig(hex_obj.id, outline=SELECTED_OUTLINE, width=3)
        for action in move_actions:
            target = self._ensure_hex(action.end_x, action.end_y)
            self.hexes.highlight(target)

    def _on_select_hand_piece(self, piece):
        self.piece_place_mode = False
        self.move_mode = False
        self.hexes.clear_highlighted()

        if not self.state.available_actions.can_be_placed(piece):
            return
        self.piece_place_mode = True
        self.piece_to_place = piece
        for spot in self.state.available_actions.place_action.spots:
            target = self._ensure_hex(spot[0], spot[1])
            self.hexes.highlight(target)

    def _try_place_piece(self, hex_obj):
        self._clear_modes()
        if not self.state.available_actions.can_be_placed_at(self.piece_to_place, hex_obj.x, hex_obj.y):
            return
        self._do_action(self.state.available_actions.place_action,
                        self.piece_to_place, hex_obj.x, hex_obj.y)

    def _try_move_piece(self, hex_obj):
        self._clear_modes()
        action = self.state.available_actions.get_move_action(
            self.hex_to_move.x, self.hex_to_move.y, hex_obj.x, hex_obj.y)
        if action is not None:
            self._do_action(action)

    def _clear_modes(self):
        self.move_mode = False
        self.piece_place_mode = False
        self.hexes.clear_highlighted()

    def _do_action(self, action, *action_args):
        self.state = action.do(self.state, *action_args)
        self.canvas.delete("all")
        self.hexes = Hexes()
        self._draw_board()
        self.hand_display.update_hand_display(
            self.state.player1_hand, self.state.player2_hand, self.state.player)
        self.info_display.update(
            self.state.player, self.state.turn, self.state.terminal, self.state.terminal_value)

    def _setup_window(self):
        window = tk.Tk()
        window.title("Hive")
        window.resizable(width=False, height=False)

        frame = tk.Frame(window)
        frame.pack()
        self.info_display = InfoDisplay(frame)
        self.hand_display = HandDisplay(frame, self._on_select_hand_piece)

        self._setup_board_canvas(frame)
        return window

    def _setup_board_canvas(self, frame):
        self.canvas = tk.Canvas(frame, width=600, height=550, bg='white')
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_canvas_click)


if __name__ == "__main__":
    GUI()
