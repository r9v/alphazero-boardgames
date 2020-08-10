import numpy as np
from const import *
import copy


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
        return (not self.placeAction.spots or not self.placeAction.pieces) and not self.moveActions

    def addPlaceAction(self, pieces, spots):
        self.placeAction = PlaceAction(pieces, spots)

    def addMoveAction(self, startX, startY, endX, endY):
        self.moveActions.append(MoveAction(startX, startY, endX, endY))

    def canBePlaced(self, piece):
        piecePresent = [
            action for action in self.placeAction.pieces if action == piece]
        return self.placeAction.spots and piecePresent

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


class GameState():
    def __init__(self, board=np.zeros((23, 23), dtype="int"), player=-1, player1Hand=Hand(), player2Hand=Hand(), turn=1):
        self.board = board
        self.player = player
        self.player1Hand = player1Hand
        self.player2Hand = player2Hand
        self.turn = turn
        self.update()

    def update(self):
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
                [Player1A, Player1G, Player1S, Player1B], [[11, 11]])
            return avalilableActions
        if self.turn == 2:  # second move close to orgin and not queen
            avalilableActions.addPlaceAction(
                [Player2A, Player2G, Player2S, Player2B], [[11, 10], [12, 10], [12, 11], [11, 12], [10, 12], [10, 11]])
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
        avalilableActions.addMoveAction(11, 11, 12, 12)

    def _getAvailablePlaceSpots(self):
        return [[11, 11]]


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
        state.board[x][y] = piece
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
        state.board[self.endX][self.endY] = state.board[self.startX][self.startY]
        state.board[self.startX][self.startY] = 0
        state.player *= -1
        state.turn += 1
        state.update()
        return state


class Hive():
    def newGame(self):
        return GameState()
