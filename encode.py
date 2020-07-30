def encode_state(board, currentPlayer):
    encoded = np.zeros([2, 6, 7]).astype(int)
    for row in range(6):
        for col in range(7):
            if board[row, col] != 0:
                encoded[{-1: 0, 1: 1}[board[row, col]], row, col] = 1
    encoded = list(np.reshape(encoded, 2*6*7))
    if currentPlayer == 1:
        encoded.append(1)
    return encoded


def decode_state(encoded):
    currentPlayer = encoded[-1]
    encoded = encoded[:-1]

    fisrt_player = encoded[:len(encoded)//2]
    second_player = encoded[len(encoded)//2:]
    board = -np.reshape(fisrt_player, ((6, 7))) + \
        np.reshape(second_player, ((6, 7)))
    return board, currentPlayer


game = Connect4Game()
game.step(0)
game.step(1)
game.step(6)
encoded = encode_state(game.board, game.currentPlayer)
board, currentPlayer = decode_state(encoded)
print(decoded)
