import numpy as np
import copy
import queue

from .pieces import *
from ..base import GameState as BaseGameState, Game


def stack_size_and_top_piece(x, y, board):
    size = 0
    piece = None
    if x < 0 or x > 24 or y < 0 or y > 24:
        return size, piece
    for z in range(5):
        if board[x][y][z] != 0:
            size = z + 1
            piece = board[x][y][z]
        else:
            break
    return size, piece


class Hand:
    def __init__(self):
        self.a = 3
        self.g = 3
        self.s = 2
        self.b = 2
        self.q = 1


class AvailableActions:
    def __init__(self):
        self.place_action = PlaceAction([], [])
        self.move_actions = []

    def empty(self):
        return (len(self.place_action.spots) == 0 or len(self.place_action.pieces) == 0) and len(self.move_actions) == 0

    def add_place_action(self, pieces, spots):
        self.place_action = PlaceAction(pieces, spots)

    def add_move_action(self, start_x, start_y, end_x, end_y):
        self.move_actions.append(MoveAction(start_x, start_y, end_x, end_y))

    def can_be_placed(self, piece):
        piece_present = [a for a in self.place_action.pieces if a == piece]
        return len(self.place_action.spots) != 0 and len(piece_present) != 0

    def get_move_actions_by_start(self, start_x, start_y):
        return [a for a in self.move_actions if a.start_x == start_x and a.start_y == start_y]

    def can_be_placed_at(self, piece, x, y):
        piece_present = [a for a in self.place_action.pieces if a == piece]
        spot_present = [a for a in self.place_action.spots if a[0] == x and a[1] == y]
        return spot_present and piece_present

    def get_move_action(self, start_x, start_y, end_x, end_y):
        actions = [a for a in self.move_actions if a.start_x == start_x
                   and a.start_y == start_y and a.end_x == end_x and a.end_y == end_y]
        if len(actions) == 0:
            return None
        return actions[0]


def neighbours(x, y):
    return [[x, y-1], [x+1, y-1], [x+1, y], [x, y+1], [x-1, y+1], [x-1, y]]


def neighbours_with_right_left(x, y):
    return [{'n': [x, y-1], 'l': [x-1, y], 'r': [x+1, y-1]},
            {'n': [x+1, y-1], 'l': [x, y-1], 'r': [x+1, y]},
            {'n': [x+1, y], 'l': [x+1, y-1], 'r': [x, y+1]},
            {'n': [x, y+1], 'l': [x+1, y], 'r': [x-1, y+1]},
            {'n': [x-1, y+1], 'l': [x, y+1], 'r': [x-1, y]},
            {'n': [x-1, y], 'l': [x-1, y+1], 'r': [x, y-1]}]


def get_player_pieces_on_top(player, board):
    player_pieces = []
    if player == -1:
        for x in range(25):
            for y in range(25):
                stack_size, piece = stack_size_and_top_piece(x, y, board)
                if piece is not None and 10 < piece < 20:
                    player_pieces.append([x, y, stack_size - 1])
    else:
        for x in range(25):
            for y in range(25):
                stack_size, piece = stack_size_and_top_piece(x, y, board)
                if piece is not None and piece > 20:
                    player_pieces.append([x, y, stack_size - 1])
    return player_pieces


def _slide_recursive(x, y, n, board, movements):
    if n == 0:
        return
    for neighbour in neighbours_with_right_left(x, y):
        nb = neighbour['n']
        r = neighbour['r']
        l = neighbour['l']
        if nb in movements:
            continue
        ss, _ = stack_size_and_top_piece(nb[0], nb[1], board)
        if ss != 0:
            continue
        ss_l, _ = stack_size_and_top_piece(l[0], l[1], board)
        ss_r, _ = stack_size_and_top_piece(r[0], r[1], board)
        if (ss_l != 0) == (ss_r != 0):
            continue
        movements.append(nb)
        _slide_recursive(nb[0], nb[1], n - 1, board, movements)


def slide(x, y, n, board):
    movements = [[x, y]]
    save = board[x][y].copy()
    board[x][y] = 0
    _slide_recursive(x, y, n, board, movements)
    movements.pop(0)
    board[x][y] = save
    return movements


def get_a_piece(board):
    for x in range(25):
        for y in range(25):
            if board[x][y][0] != 0:
                return [x, y]


def hive_broken(board, n_all_pieces):
    que = queue.Queue(22)
    piece = get_a_piece(board)
    part_of_hive = np.zeros((25, 25), dtype=bool)
    part_of_hive[piece[0]][piece[1]] = True
    n_visited, _ = stack_size_and_top_piece(piece[0], piece[1], board)
    que.put(piece)
    while not que.empty():
        piece = que.get()
        for n in neighbours(piece[0], piece[1]):
            ss, _ = stack_size_and_top_piece(n[0], n[1], board)
            if ss == 0:
                continue
            if part_of_hive[n[0]][n[1]]:
                continue
            n_visited += ss
            part_of_hive[n[0]][n[1]] = True
            que.put([n[0], n[1]])
    return n_all_pieces - 1 != n_visited


def move_breaks_hive(piece, board, n_all_pieces):
    if piece[2] > 0:
        return False
    ns = neighbours(piece[0], piece[1])
    ss, _ = stack_size_and_top_piece(ns[0][0], ns[0][1], board)
    last_has_piece = ss > 0
    it_ns = iter(ns)
    next(it_ns)
    edges = 0
    for n in it_ns:
        ss, _ = stack_size_and_top_piece(n[0], n[1], board)
        has_piece = ss > 0
        if last_has_piece != has_piece:
            edges += 1
            if edges > 2:
                break
        last_has_piece = has_piece
    if edges <= 2:
        return False
    save = board[piece[0]][piece[1]][piece[2]]
    board[piece[0]][piece[1]][piece[2]] = 0
    broken = hive_broken(board, n_all_pieces)
    board[piece[0]][piece[1]][piece[2]] = save
    return broken


def ant_movement(x, y, board):
    return slide(x, y, 9999999, board)


def queen_movement(x, y, board):
    return slide(x, y, 1, board)


def grass_movement(x, y, board):
    movements = []
    for idx, n in enumerate(neighbours(x, y)):
        distance = 0
        ss, piece = stack_size_and_top_piece(n[0], n[1], board)
        while ss:
            n = neighbours(n[0], n[1])[idx]
            ss, piece = stack_size_and_top_piece(n[0], n[1], board)
            distance += 1
        if distance > 0:
            movements.append(n)
    return movements


def spider_movement(x, y, board):
    two = slide(x, y, 2, board)
    if not two:
        return []
    three = slide(x, y, 3, board)
    if not three:
        return []
    return [i for i in three if i not in two]


def beetle_movement(x, y, board):
    movements = []
    for neighbour in neighbours_with_right_left(x, y):
        n = neighbour['n']
        r = neighbour['r']
        l = neighbour['l']
        ss_n, _ = stack_size_and_top_piece(n[0], n[1], board)
        ss_l, _ = stack_size_and_top_piece(l[0], l[1], board)
        ss_r, _ = stack_size_and_top_piece(r[0], r[1], board)
        if not ss_n and not ss_l and not ss_r:
            continue
        if ss_n < ss_l and ss_n < ss_r:
            ss_b, _ = stack_size_and_top_piece(x, y, board)
            ss_b -= 1
            if ss_b < ss_l and ss_b < ss_r:
                continue
        movements.append(n)
    return movements


def board_center(board):
    board2 = board[:, :, 0].copy()
    min_x = min_y = 24
    max_x = max_y = 0
    for x in range(25):
        for y in range(25):
            if board2[x][y] == 0:
                continue
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
    return (max_x + min_x) / 2.0, (max_y + min_y) / 2.0


class GameState(BaseGameState):
    def __init__(self, board=None, player=-1, player1_hand=None, player2_hand=None, turn=1):
        if board is None:
            board = np.zeros((25, 25, 5), dtype="int")
        if player1_hand is None:
            player1_hand = Hand()
        if player2_hand is None:
            player2_hand = Hand()
        self.board = board
        self.player = player
        self.player1_hand = player1_hand
        self.player2_hand = player2_hand
        self.turn = turn
        self.prev_state = None
        self.last_turn_skipped = False
        self._update()

    def _update(self):
        cx, cy = board_center(self.board)
        xshift = int(round(12 - cx))
        yshift = int(round(12 - cy))
        if xshift:
            self.board = np.roll(self.board, xshift, axis=0)
        if yshift:
            self.board = np.roll(self.board, yshift, axis=1)

        self.available_actions = self._compute_available_actions()
        self.terminal, self.terminal_value = self._over()

        if self.available_actions.empty() and not self.terminal:
            self.turn += 1
            self.player *= -1
            self.last_turn_skipped = True
            self.available_actions = self._compute_available_actions()

    def _over(self):
        return False, None

    def _compute_available_actions(self):
        aa = AvailableActions()
        if self.turn == 1:
            aa.add_place_action(
                [Player1A, Player1G, Player1S, Player1B], [[12, 12]])
            return aa
        if self.turn == 2:
            aa.add_place_action(
                [Player2A, Player2G, Player2S, Player2B], neighbours(12, 12))
            return aa

        if self.turn > 6 and self.turn < 9:
            if self.player == -1 and self.player1_hand.q == 1:
                aa.add_place_action([Player1Q], self._get_available_place_spots())
                return aa
            if self.player == 1 and self.player2_hand.q == 1:
                aa.add_place_action([Player2Q], self._get_available_place_spots())
                return aa

        pieces_to_place = self._get_pieces_to_place()
        if pieces_to_place:
            aa.add_place_action(pieces_to_place, self._get_available_place_spots())
        self._add_move_actions(aa)
        return aa

    def _get_pieces_to_place(self):
        pieces = []
        if self.player == -1:
            hand = self.player1_hand
            constants = [Player1A, Player1G, Player1S, Player1B, Player1Q]
        else:
            hand = self.player2_hand
            constants = [Player2A, Player2G, Player2S, Player2B, Player2Q]
        for piece_type, count in zip(constants, [hand.a, hand.g, hand.s, hand.b, hand.q]):
            if count > 0:
                pieces.append(piece_type)
        return pieces

    def _add_move_actions(self, aa):
        player_pieces = get_player_pieces_on_top(self.player, self.board)
        n_all = np.count_nonzero(self.board)
        for piece in player_pieces:
            if move_breaks_hive(piece, self.board, n_all):
                continue
            p = self.board[piece[0]][piece[1]][piece[2]]
            if p in (Player1A, Player2A):
                movements = ant_movement(piece[0], piece[1], self.board)
            elif p in (Player1G, Player2G):
                movements = grass_movement(piece[0], piece[1], self.board)
            elif p in (Player1S, Player2S):
                movements = spider_movement(piece[0], piece[1], self.board)
            elif p in (Player1B, Player2B):
                movements = beetle_movement(piece[0], piece[1], self.board)
            elif p in (Player1Q, Player2Q):
                movements = queen_movement(piece[0], piece[1], self.board)
            else:
                movements = []
            for m in movements:
                aa.add_move_action(piece[0], piece[1], m[0], m[1])

    def _get_available_place_spots(self):
        spots = []
        player_pieces = get_player_pieces_on_top(self.player, self.board)
        for piece_xyz in player_pieces:
            for neighbour in neighbours(piece_xyz[0], piece_xyz[1]):
                available = True
                if self.board[neighbour[0]][neighbour[1]][0] == 0:
                    for neighbour2 in neighbours(neighbour[0], neighbour[1]):
                        if self.player == -1:
                            _, top = stack_size_and_top_piece(
                                neighbour2[0], neighbour2[1], self.board)
                            enemy = top is not None and top > 20
                        else:
                            _, top = stack_size_and_top_piece(
                                neighbour2[0], neighbour2[1], self.board)
                            enemy = top is not None and 10 < top < 20
                        if enemy:
                            available = False
                            break
                else:
                    available = False
                if available and neighbour not in spots:
                    spots.append(neighbour)
        return spots


class PlaceAction:
    def __init__(self, pieces, spots):
        self.pieces = pieces
        self.spots = spots

    def do(self, state, piece, x, y):
        state = copy.deepcopy(state)
        if piece == Player1A:
            state.player1_hand.a -= 1
        elif piece == Player1G:
            state.player1_hand.g -= 1
        elif piece == Player1S:
            state.player1_hand.s -= 1
        elif piece == Player1B:
            state.player1_hand.b -= 1
        elif piece == Player1Q:
            state.player1_hand.q -= 1
        elif piece == Player2A:
            state.player2_hand.a -= 1
        elif piece == Player2G:
            state.player2_hand.g -= 1
        elif piece == Player2S:
            state.player2_hand.s -= 1
        elif piece == Player2B:
            state.player2_hand.b -= 1
        elif piece == Player2Q:
            state.player2_hand.q -= 1
        state.board[x][y][0] = piece
        state.player *= -1
        state.turn += 1
        state._update()
        return state


class MoveAction:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def do(self, state):
        state = copy.deepcopy(state)
        ss_a, piece_a = stack_size_and_top_piece(
            self.start_x, self.start_y, state.board)
        ss_b, _ = stack_size_and_top_piece(
            self.end_x, self.end_y, state.board)
        state.board[self.end_x][self.end_y][ss_b] = piece_a
        state.board[self.start_x][self.start_y][ss_a - 1] = 0
        state.player *= -1
        state.turn += 1
        state._update()
        return state


class HiveGame(Game):
    board_shape = (25, 25)
    action_size = None  # Hive's action space is complex; flat encoding TBD
    num_history_states = 0

    def new_game(self):
        return GameState()

    def step(self, state, action):
        raise NotImplementedError(
            "Hive uses PlaceAction/MoveAction.do() directly. "
            "Flat action encoding not yet implemented."
        )

    def state_to_input(self, state):
        raise NotImplementedError(
            "Hive neural network input encoding not yet implemented."
        )
