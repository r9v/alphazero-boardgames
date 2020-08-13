import numpy as np
from const import *
import copy
import queue


def stackSizeAndTopPiece(x, y, board):
    size = 0
    piece = None
    if x < 0 or x > 24 or y < 0 or y > 24:
        return size, piece
    for z in range(5):
        if board[x][y][z] != 0:
            size = z+1
            piece = board[x][y][z]
        else:
            break
    return size, piece


class Hand():
    def __init__(self):
        self.a = 3
        self.g = 3
        self.s = 2
        self.b = 2
        self.q = 1


class AvalilableActions():
    def __init__(self):
        self.placeAction = PlaceAction([], [])
        self.moveActions = []

    def empty(self):
        return (len(self.placeAction.spots) == 0 or len(self.placeAction.pieces) == 0) and len(self.moveActions) == 0

    def addPlaceAction(self, pieces, spots):
        self.placeAction = PlaceAction(pieces, spots)

    def addMoveAction(self, startX, startY, endX, endY):
        self.moveActions.append(MoveAction(startX, startY, endX, endY))

    def canBePlaced(self, piece):
        piecePresent = [
            action for action in self.placeAction.pieces if action == piece]
        return len(self.placeAction.spots) != 0 and len(piecePresent) != 0

    def getMoveActionsByStart(self, startX, startY):
        return [action for action in self.moveActions if action.startX == startX and action.startY == startY]

    def canBePlacedAt(self, piece, x, y):
        piecePresent = [
            action for action in self.placeAction.pieces if action == piece]
        spotPresent = [
            action for action in self.placeAction.spots if action[0] == x and action[1] == y]
        return spotPresent and piecePresent

        if len(actions) == 0:
            return None
        return actions[0]

    def getMoveAction(self, startX, startY, endX, endY):
        actions = [action for action in self.moveActions if action.startX == startX
                   and action.startY == startY and action.endX == endX and action.endY == endY]
        if len(actions) == 0:
            return None
        return actions[0]


def neighbours(x, y):
    return [[x, y-1], [x+1, y-1], [x+1, y], [x, y+1], [x-1, y+1], [x-1, y]]


def neighboursWithRightLeft(x, y):
    return [{'n': [x, y-1], 'l':[x-1, y], 'r':[x+1, y-1]},
            {'n': [x+1, y-1], 'l':[x, y-1], 'r':[x+1, y]},
            {'n': [x+1, y], 'l':[x+1, y-1], 'r':[x, y+1]},
            {'n': [x, y+1], 'l':[x+1, y], 'r':[x-1, y+1]},
            {'n': [x-1, y+1], 'l':[x, y+1], 'r':[x-1, y]},
            {'n': [x-1, y], 'l':[x-1, y+1], 'r':[x, y-1]}]


def getPlayerPiecesIfOnTopXYZ(player, board):
    playerPieces = []
    if player == -1:
        for x in range(25):
            for y in range(25):
                stackSize, piece = stackSizeAndTopPiece(x, y, board)
                if piece is not None and piece > 10 and piece < 20:
                    playerPieces.append([x, y, stackSize-1])
    else:
        for x in range(25):
            for y in range(25):
                stackSize, piece = stackSizeAndTopPiece(x, y, board)
                if piece is not None and piece > 20:
                    playerPieces.append([x, y, stackSize-1])
    return playerPieces


def slide2(x, y, N, board, movements):
    if N == 0:
        return
    for neighbour in neighboursWithRightLeft(x, y):
        n = neighbour['n']
        r = neighbour['r']
        l = neighbour['l']
        if n in movements:
            continue
        stackSize, _ = stackSizeAndTopPiece(n[0], n[1], board)
        if stackSize != 0:
            continue
        stackSizeL, _ = stackSizeAndTopPiece(l[0], l[1], board)
        stackSizeR, _ = stackSizeAndTopPiece(r[0], r[1], board)
        if (stackSizeL != 0) == (stackSizeR != 0):
            continue
        movements.append(n)
        slide2(n[0], n[1], N-1, board, movements)
    return


def slide(x, y, N, board):
    movements = [[x, y]]
    save = board[x][y].copy()
    board[x][y] = 0
    slide2(x, y, N, board, movements)
    movements.pop(0)
    board[x][y] = save
    return movements


def getAPiece(board):
    for x in range(25):
        for y in range(25):
            if board[x][y][0] != 0:
                return [x, y]


def hiveBroken(board, nAllPieces):
    que = queue.Queue(22)
    piece = getAPiece(board)
    partOfHive = np.zeros((25, 25), dtype=bool)
    partOfHive[piece[0]][piece[1]] = True
    nVisited, _ = stackSizeAndTopPiece(piece[0], piece[1], board)
    que.put(piece)
    while not que.empty():
        piece = que.get()
        for n in neighbours(piece[0], piece[1]):
            stackSize, _ = stackSizeAndTopPiece(n[0], n[1], board)
            if stackSize == 0:
                continue
            if partOfHive[n[0]][n[1]]:
                continue
            nVisited += stackSize
            partOfHive[n[0]][n[1]] = True
            que.put([n[0], n[1]])
    return nAllPieces-1 != nVisited


def moveBreakesHive(piece, board, nAllPieces):
    if piece[2] > 0:
        return False
    ns = neighbours(piece[0], piece[1])
    stackSize, _ = stackSizeAndTopPiece(ns[0][0], ns[0][1], board)
    lastHasPiece = stackSize > 0
    iterns = iter(ns)
    next(iterns)
    edges = 0
    for n in iterns:
        stackSize, _ = stackSizeAndTopPiece(n[0], n[1], board)
        hasPiece = stackSize > 0
        if lastHasPiece != hasPiece:
            edges += 1
            if edges > 2:
                break
        lastHasPiece = hasPiece
    if edges <= 2:
        return False
    save = board[piece[0]][piece[1]][piece[2]]
    board[piece[0]][piece[1]][piece[2]] = 0
    broken = hiveBroken(board, nAllPieces)
    board[piece[0]][piece[1]][piece[2]] = save
    return broken


def antMovement(x, y, board):
    return slide(x, y, 9999999, board)


def queenMovement(x, y, board):
    return slide(x, y, 1, board)


def grassMovement(x, y, board):
    movements = []
    for idx, n in enumerate(neighbours(x, y)):
        distance = 0
        stackSize, piece = stackSizeAndTopPiece(n[0], n[1], board)
        while stackSize:
            n = neighbours(n[0], n[1])[idx]
            stackSize, piece = stackSizeAndTopPiece(n[0], n[1], board)
            distance += 1
        if distance > 0:
            movements.append(n)
    return movements


def spiderMovement(x, y, board):
    two = slide(x, y, 2, board)
    if not two:
        return []
    three = slide(x, y, 3, board)
    if not three:
        return []
    return [i for i in three if i not in two]


def beetleMovement(x, y, board):
    movements = []
    for neighbour in neighboursWithRightLeft(x, y):
        n = neighbour['n']
        r = neighbour['r']
        l = neighbour['l']
        stackSizeN, piece = stackSizeAndTopPiece(n[0], n[1], board)
        stackSizeL, piece = stackSizeAndTopPiece(l[0], l[1], board)
        stackSizeR, piece = stackSizeAndTopPiece(r[0], r[1], board)
        if not stackSizeN and not stackSizeL and not stackSizeR:
            continue
        if stackSizeN < stackSizeL and stackSizeN < stackSizeR:
            stackSizeB, piece = stackSizeAndTopPiece(x, y, board)
            stackSizeB -= 1
            if stackSizeB < stackSizeL and stackSizeB < stackSizeR:
                continue
        movements.append(n)
    return movements


def boardCenter(board):
    board2 = board[:, :, 0].copy()
    minx = miny = 24
    maxx = maxy = 0
    for x in range(25):
        for y in range(25):
            if board2[x][y] == 0:
                continue
            if x < minx:
                minx = x
            if y < miny:
                miny = y
            if x > maxx:
                maxx = x
            if y > maxy:
                maxy = y
    return (maxx+minx)/2., (maxy+miny)/2.


class GameState():
    def __init__(self, board=np.zeros((25, 25, 5), dtype="int"), player=-1, player1Hand=Hand(), player2Hand=Hand(), turn=1):
        self.board = board
        self.player = player
        self.player1Hand = player1Hand
        self.player2Hand = player2Hand
        self.turn = turn
        self.update()

    def update(self):
        cx, cy = boardCenter(self.board)
        xshift = int(round(12-cx))
        yshift = int(round(12-cy))
        if xshift:
            self.board = np.roll(self.board, xshift,  axis=0)
        if yshift:
            self.board = np.roll(self.board, yshift,  axis=1)
        self.availableActions = self._availableActions()
        self.terminal, self.terminalValue = self._over()

        if self.availableActions.empty() and not self.terminal:  # if player blocked this turn
            self.turn += 1
            self.player *= -1
            self.availableActions = self._availableActions()

    def _over(self):
        return False, None

    def _availableActions(self):
        avalilableActions = AvalilableActions()
        if self.turn == 1:  # first move at orgin and not queen
            avalilableActions.addPlaceAction(
                [Player1A, Player1G, Player1S, Player1B], [[12, 12]])
            return avalilableActions
        if self.turn == 2:  # second move close to orgin and not queen
            avalilableActions.addPlaceAction(
                [Player2A, Player2G, Player2S, Player2B], neighbours(12, 12))
            return avalilableActions

        if self.turn > 6 and self.turn < 9:
            if self.player == -1 and self.player1Hand.q == 1:
                avalilableActions.addPlaceAction(
                    [Player1Q], self._getAvailablePlaceSpots())
                return avalilableActions
            if self.player == 1 and self.player2Hand.q == 1:
                avalilableActions.addPlaceAction(
                    [Player2Q], self._getAvailablePlaceSpots())
                return avalilableActions

        piecesToPlace = self._getPiecesToPlace()
        if piecesToPlace:
            avalilableActions.addPlaceAction(
                piecesToPlace,  self._getAvailablePlaceSpots())
        self._addMoveActions(avalilableActions)
        return avalilableActions

    def _getPiecesToPlace(self):
        pieces = []
        if(self.player == -1):
            if(self.player1Hand.a > 0):
                pieces.append(Player1A)
            if(self.player1Hand.g > 0):
                pieces.append(Player1G)
            if(self.player1Hand.s > 0):
                pieces.append(Player1S)
            if(self.player1Hand.b > 0):
                pieces.append(Player1B)
            if(self.player1Hand.q > 0):
                pieces.append(Player1Q)
            return pieces
        if(self.player2Hand.a > 0):
            pieces.append(Player2A)
        if(self.player2Hand.g > 0):
            pieces.append(Player2G)
        if(self.player2Hand.s > 0):
            pieces.append(Player2S)
        if(self.player2Hand.b > 0):
            pieces.append(Player2B)
        if(self.player2Hand.q > 0):
            pieces.append(Player2Q)
        return pieces

    def _addMoveActions(self, avalilableActions):
        playerPiecesOnTopXYZ = getPlayerPiecesIfOnTopXYZ(
            self.player, self.board)
        nAllPieces = np.count_nonzero(self.board)
        for piece in playerPiecesOnTopXYZ:
            if moveBreakesHive(piece, self.board, nAllPieces):
                continue
            movements = []
            if self.board[piece[0]][piece[1]][piece[2]] == Player1A or self.board[piece[0]][piece[1]][piece[2]] == Player2A:
                movements = antMovement(piece[0], piece[1], self.board)
            elif self.board[piece[0]][piece[1]][piece[2]] == Player1G or self.board[piece[0]][piece[1]][piece[2]] == Player2G:
                movements = grassMovement(piece[0], piece[1], self.board)
            elif self.board[piece[0]][piece[1]][piece[2]] == Player1S or self.board[piece[0]][piece[1]][piece[2]] == Player2S:
                movements = spiderMovement(piece[0], piece[1], self.board)
            elif self.board[piece[0]][piece[1]][piece[2]] == Player1B or self.board[piece[0]][piece[1]][piece[2]] == Player2B:
                movements = beetleMovement(piece[0], piece[1], self.board)
            elif self.board[piece[0]][piece[1]][piece[2]] == Player1Q or self.board[piece[0]][piece[1]][piece[2]] == Player2Q:
                movements = queenMovement(piece[0], piece[1], self.board)
            for movement in movements:
                avalilableActions.addMoveAction(
                    piece[0], piece[1], movement[0], movement[1])

    def _getAvailablePlaceSpots(self):
        # print(np.swapaxes(self.board, 0, 1))
        spots = []
        playerPiecesOnTopXYZ = getPlayerPiecesIfOnTopXYZ(
            self.player, self.board)

        for pieceXYZ in playerPiecesOnTopXYZ:
            for neighbour in neighbours(pieceXYZ[0], pieceXYZ[1]):
                available = True
                # check idx 25
                if self.board[neighbour[0]][neighbour[1]][0] == 0:
                    for neighbour2 in neighbours(neighbour[0], neighbour[1]):
                        enemyPresent = False
                        if self.player == -1:
                            stackSize, topPiece = stackSizeAndTopPiece(
                                neighbour2[0], neighbour2[1], self.board)
                            enemyPresent = topPiece is not None and topPiece > 20
                        else:
                            stackSize, topPiece = stackSizeAndTopPiece(
                                neighbour2[0], neighbour2[1], self.board)
                            enemyPresent = topPiece is not None and topPiece > 10 and topPiece < 20
                        if enemyPresent:
                            available = False
                            break
                else:
                    available = False
                if available and neighbour not in spots:
                    spots.append(neighbour)
        return spots


class PlaceAction():
    def __init__(self, pieces, spots):
        self.pieces = pieces
        self.spots = spots

    def do(self, state: GameState, piece, x, y):
        state = copy.deepcopy(state)
        if(piece == Player1A):
            state.player1Hand.a -= 1
        elif(piece == Player1G):
            state.player1Hand.g -= 1
        elif(piece == Player1S):
            state.player1Hand.s -= 1
        elif(piece == Player1B):
            state.player1Hand.b -= 1
        elif(piece == Player1Q):
            state.player1Hand.q -= 1
        elif(piece == Player2A):
            state.player2Hand.a -= 1
        elif(piece == Player2G):
            state.player2Hand.g -= 1
        elif(piece == Player2S):
            state.player2Hand.s -= 1
        elif(piece == Player2B):
            state.player2Hand.b -= 1
        elif(piece == Player2Q):
            state.player2Hand.q -= 1
        state.board[x][y][0] = piece
        state.player *= -1
        state.turn += 1
        state.update()
        return state


class MoveAction():
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

    def do(self, state: GameState):
        state = copy.deepcopy(state)
        stackSizeA, pieceA = stackSizeAndTopPiece(
            self.startX, self.startY, state.board)
        stackSizeB, pieceB = stackSizeAndTopPiece(
            self.endX, self.endY, state.board)

        state.board[self.endX][self.endY][stackSizeB] = pieceA
        state.board[self.startX][self.startY][stackSizeA-1] = 0
        state.player *= -1
        state.turn += 1
        state.update()
        return state


class Hive():
    def newGame(self):
        return GameState()
