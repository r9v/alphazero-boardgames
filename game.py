import numpy as np

ROW_COUNT = 6
COLUMN_COUNT = 7


class Game():
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype="int")
        self.current_player = -1

    def step(self, column):
        if(column < 0 or column > COLUMN_COUNT-1):
            raise Exception(f'Invalid column {column}')

        row = self.get_free_row(column)
        if row is None:
            raise Exception(f'Invalid action, column {column} is full')
        self.board[row][column] = self.current_player
        self.current_player *= -1

        score = self.check_game_over()
        if score is not None:
            return True, score
        return False, 0

    def get_free_row(self, column):
        for row in range(ROW_COUNT):
            if self.board[row][column] == 0:
                return row
        return None

    def available_moves(self):
        available_moves = []
        for column in range(COLUMN_COUNT):
            if self.get_free_row(column) is not None:
                available_moves.append(column)
        return available_moves

    def print(self):
        print(np.flip(self.board, 0))

    def check_player_won(self, player):
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT):
                if self.board[r][c] == player and self.board[r][c+1] == player and self.board[r][c+2] == player and self.board[r][c+3] == player:
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if self.board[r][c] == player and self.board[r+1][c] == player and self.board[r+2][c] == player and self.board[r+3][c] == player:
                    return True

        # Check positively sloped diaganols
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT-3):
                if self.board[r][c] == player and self.board[r+1][c+1] == player and self.board[r+2][c+2] == player and self.board[r+3][c+3] == player:
                    return True

        # Check negatively sloped diaganols
        for c in range(COLUMN_COUNT-3):
            for r in range(3, ROW_COUNT):
                if self.board[r][c] == player and self.board[r-1][c+1] == player and self.board[r-2][c+2] == player and self.board[r-3][c+3] == player:
                    return True

    def check_game_over(self):
        if np.count_nonzero(self.board) == ROW_COUNT*COLUMN_COUNT:
            return 0
        if(self.check_player_won(-1)):
            return -1
        if(self.check_player_won(1)):
            return 1
        return None
