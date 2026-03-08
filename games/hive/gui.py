import tkinter as tk
import math
from .game import HiveGame, stack_size_and_top_piece
from .pieces import *

hive = HiveGame()


class InfoDisplay:
    def __init__(self, frame):
        self.player = tk.StringVar()
        self.turn = tk.StringVar()
        self.terminal = tk.StringVar()
        self.terminal_value = tk.StringVar()
        frame = tk.Frame(frame)
        tk.Label(frame, textvariable=self.player).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.turn).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.terminal).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.terminal_value).pack(side=tk.LEFT)
        frame.pack()

    def update(self, player, turn, terminal, terminal_value):
        self.player.set(f'player: {player}, ')
        self.turn.set(f'turn: {turn}, ')
        self.terminal.set(f'terminal: {terminal}, ')
        self.terminal_value.set(f'terminalValue: {terminal_value}, ')


class HandDisplay:
    def __init__(self, frame, on_button_click):
        left_frame = tk.Frame(frame)
        right_frame = tk.Frame(frame)
        left_frame.pack(side=tk.LEFT)
        right_frame.pack(side=tk.RIGHT)

        self.p1a_button = tk.Button(left_frame, width=10, command=lambda: on_button_click(Player1A))
        self.p1g_button = tk.Button(left_frame, width=10, command=lambda: on_button_click(Player1G))
        self.p1s_button = tk.Button(left_frame, width=10, command=lambda: on_button_click(Player1S))
        self.p1b_button = tk.Button(left_frame, width=10, command=lambda: on_button_click(Player1B))
        self.p1q_button = tk.Button(left_frame, width=10, command=lambda: on_button_click(Player1Q))

        self.p2a_button = tk.Button(right_frame, width=10, command=lambda: on_button_click(Player2A))
        self.p2g_button = tk.Button(right_frame, width=10, command=lambda: on_button_click(Player2G))
        self.p2s_button = tk.Button(right_frame, width=10, command=lambda: on_button_click(Player2S))
        self.p2b_button = tk.Button(right_frame, width=10, command=lambda: on_button_click(Player2B))
        self.p2q_button = tk.Button(right_frame, width=10, command=lambda: on_button_click(Player2Q))

        for btn in [self.p1a_button, self.p1g_button, self.p1s_button, self.p1b_button, self.p1q_button]:
            btn.pack(side=tk.TOP)
        for btn in [self.p2a_button, self.p2g_button, self.p2s_button, self.p2b_button, self.p2q_button]:
            btn.pack(side=tk.TOP)

    def update_hand_display(self, player1_hand, player2_hand):
        self.p1a_button['text'] = f'Ant {player1_hand.a}'
        self.p1g_button['text'] = f'Grass {player1_hand.g}'
        self.p1s_button['text'] = f'Spider {player1_hand.s}'
        self.p1b_button['text'] = f'Beetle {player1_hand.b}'
        self.p1q_button['text'] = f'Queen {player1_hand.q}'

        self.p2a_button['text'] = f'Ant {player2_hand.a}'
        self.p2g_button['text'] = f'Grass {player2_hand.g}'
        self.p2s_button['text'] = f'Spider {player2_hand.s}'
        self.p2b_button['text'] = f'Beetle {player2_hand.b}'
        self.p2q_button['text'] = f'Queen {player2_hand.q}'


class Hex:
    def __init__(self, id, x, y, canvas, canvas_x, canvas_y):
        self.id = id
        self.x = x
        self.y = y
        self.canvas = canvas
        self.canvas_x = canvas_x
        self.canvas_y = canvas_y
        self.text = None
        self.piece = None

    def highlight(self):
        self.canvas.itemconfig(self.id, outline='blue')
        self.canvas.tag_raise(self.id)

    def clear_highlight(self):
        self.canvas.itemconfig(self.id, outline='black')

    def set_piece(self, piece, stack_size):
        text = piece
        if stack_size != 1:
            text = f'{stack_size} {piece}'
        self.text = self.canvas.create_text(self.canvas_x, self.canvas_y, text=text)
        self.piece = piece


class Hexes:
    def __init__(self):
        self.hexes = []
        self.highlighted = []

    def add(self, hex):
        self.hexes.append(hex)

    def get_by_id(self, id):
        return next((h for h in self.hexes if h.id == id or h.text == id), None)

    def get_by_xy(self, x, y):
        return next((h for h in self.hexes if h.x == x and h.y == y), None)

    def highlight(self, x, y):
        h = self.get_by_xy(x, y)
        h.highlight()
        self.highlighted.append(h)

    def clear_highlighted(self):
        for h in self.highlighted:
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
            self.state.player1_hand, self.state.player2_hand)
        self.info_display.update(
            self.state.player, self.state.turn, self.state.terminal, self.state.terminal_value)
        self._draw_board()

        window.mainloop()

    def _draw_board(self):
        size = 20
        yoff = 50
        xoff = 50
        h = 2 * size
        w = math.sqrt(3) * size
        for j in range(25):
            x = xoff + j * w / 2
            y = j * 3 * h / 4 + yoff
            for i in range(25):
                id = self._draw_hex(x, y, size)
                hex = Hex(id, i, j, self.canvas, x, y)
                ss, piece = stack_size_and_top_piece(i, j, self.state.board)
                if ss != 0:
                    hex.set_piece(piece, ss)
                self.hexes.add(hex)
                x += w

    def _draw_hex(self, x, y, size):
        points = []
        for i in range(6):
            angle = math.pi / 180 * (60 * i - 30)
            points.append(x + size * math.cos(angle))
            points.append(y + size * math.sin(angle))
        return self.canvas.create_polygon(points, width=3, fill="", outline='black')

    def _on_canvas_click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        hex_id = event.widget.find_overlapping(x, y, x + 1, y + 1)
        if len(hex_id) == 0:
            return
        self._on_hex_click(hex_id[0])

    def _on_hex_click(self, hex_id):
        hex = self.hexes.get_by_id(hex_id)
        if self.piece_place_mode:
            self._try_place_piece(hex)
        elif self.move_mode:
            self._try_move_piece(hex)
        elif hex.piece:
            self._start_move_action(hex)

    def _start_move_action(self, hex):
        move_actions = self.state.available_actions.get_move_actions_by_start(hex.x, hex.y)
        if len(move_actions) == 0:
            return
        self.move_mode = True
        self.hex_to_move = hex
        for action in move_actions:
            self.hexes.highlight(action.end_x, action.end_y)

    def _on_select_hand_piece(self, piece):
        self.piece_place_mode = False
        self.move_mode = False
        self.hexes.clear_highlighted()

        if not self.state.available_actions.can_be_placed(piece):
            return
        self.piece_place_mode = True
        self.piece_to_place = piece
        for spot in self.state.available_actions.place_action.spots:
            self.hexes.highlight(spot[0], spot[1])

    def _try_place_piece(self, hex):
        self._clear_modes()
        if not self.state.available_actions.can_be_placed_at(self.piece_to_place, hex.x, hex.y):
            return
        self._do_action(self.state.available_actions.place_action,
                        self.piece_to_place, hex.x, hex.y)

    def _try_move_piece(self, hex):
        self._clear_modes()
        action = self.state.available_actions.get_move_action(
            self.hex_to_move.x, self.hex_to_move.y, hex.x, hex.y)
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
            self.state.player1_hand, self.state.player2_hand)
        self.info_display.update(
            self.state.player, self.state.turn, self.state.terminal, self.state.terminal_value)

    def _setup_window(self):
        window = tk.Tk()
        window.resizable(width=False, height=False)

        frame = tk.Frame(window)
        frame.pack()
        self.info_display = InfoDisplay(frame)
        self.hand_display = HandDisplay(frame, self._on_select_hand_piece)

        self._setup_board_canvas(frame)
        return window

    def _setup_board_canvas(self, frame):
        self.canvas = tk.Canvas(frame, width=500, height=500,
                                scrollregion=(0, 0, 1220, 750), bg='white')

        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.yview_moveto(0.20)
        self.canvas.xview_moveto(0.34)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self._on_canvas_click)


if __name__ == "__main__":
    GUI()
