
class Utils:
    @staticmethod
    def encodeState(board, currentPlayer):
        encoded = np.zeros([2, 6, 7]).astype(int)
        for row in range(6):
            for col in range(7):
                if board[row, col] != 0:
                    encoded[{-1: 0, 1: 1}[board[row, col]], row, col] = 1
        encoded = list(np.reshape(encoded, 2*6*7))
        if currentPlayer == 1:
            encoded.append(1)
        return encoded

    @staticmethod
    def decodeState(encoded):
        currentPlayer = encoded[-1]
        encoded = encoded[:-1]

        firstPlayerState = encoded[:len(encoded)//2]
        secondPlayerState = encoded[len(encoded)//2:]
        board = -np.reshape(firstPlayerState, ((6, 7))) + \
            np.reshape(secondPlayerState, ((6, 7)))
        return board, currentPlayer
