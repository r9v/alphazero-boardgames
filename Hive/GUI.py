import tkinter as tk
import math
from Hive import Hive, stackSizeAndTopPiece
from const import *

hive = Hive()


class InfoDisplay():
    def __init__(self, frame):
        self.player = tk.StringVar()
        self.turn = tk.StringVar()
        self.terminal = tk.StringVar()
        self.terminalValue = tk.StringVar()
        frame = tk.Frame(frame)
        tk.Label(frame, textvariable=self.player).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.turn).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.terminal).pack(side=tk.LEFT)
        tk.Label(frame, textvariable=self.terminalValue).pack(side=tk.LEFT)
        frame.pack()

    def update(self, player, turn, terminal, terminalValue):
        self.player.set(f'player: {player}, ')
        self.turn.set(f'turn: {turn}, ')
        self.terminal.set(f'terminal: {terminal}, ')
        self.terminalValue.set(f'terminalValue: {terminalValue}, ')


class HandDisplay():
    def __init__(self, frame, onButtonClick):

        leftFrame = tk.Frame(frame)
        self.leftFrame = leftFrame
        rightFrame = tk.Frame(frame)
        leftFrame.pack(side=tk.LEFT)
        rightFrame.pack(side=tk.RIGHT)

        self.p1aButton = tk.Button(
            leftFrame, width=10, command=lambda: onButtonClick(Player1A))
        self.p1gButton = tk.Button(
            leftFrame, width=10, command=lambda: onButtonClick(Player1G))
        self.p1sButton = tk.Button(
            leftFrame, width=10, command=lambda: onButtonClick(Player1S))
        self.p1bButton = tk.Button(
            leftFrame, width=10, command=lambda: onButtonClick(Player1B))
        self.p1qButton = tk.Button(
            leftFrame, width=10, command=lambda: onButtonClick(Player1Q))

        self.p2aButton = tk.Button(
            rightFrame, width=10, command=lambda: onButtonClick(Player2A))
        self.p2gButton = tk.Button(
            rightFrame, width=10, command=lambda: onButtonClick(Player2G))
        self.p2sButton = tk.Button(
            rightFrame, width=10, command=lambda: onButtonClick(Player2S))
        self.p2bButton = tk.Button(
            rightFrame, width=10, command=lambda: onButtonClick(Player2B))
        self.p2qButton = tk.Button(
            rightFrame, width=10, command=lambda: onButtonClick(Player2Q))

        self.p1aButton.pack(side=tk.TOP)
        self.p1gButton.pack(side=tk.TOP)
        self.p1sButton.pack(side=tk.TOP)
        self.p1bButton.pack(side=tk.TOP)
        self.p1qButton.pack(side=tk.TOP)

        self.p2aButton.pack(side=tk.TOP)
        self.p2gButton.pack(side=tk.TOP)
        self.p2sButton.pack(side=tk.TOP)
        self.p2bButton.pack(side=tk.TOP)
        self.p2qButton.pack(side=tk.TOP)

    def updateHandDisplay(self, player1Hand, player2Hand):
        self.p1aButton['text'] = f'Ant {player1Hand.a}'
        self.p1gButton['text'] = f'Grass {player1Hand.g}'
        self.p1sButton['text'] = f'Spider {player1Hand.s}'
        self.p1bButton['text'] = f'Beetle {player1Hand.b}'
        self.p1qButton['text'] = f'Queen {player1Hand.q}'

        self.p2aButton['text'] = f'Ant {player2Hand.a}'
        self.p2gButton['text'] = f'Grass {player2Hand.g}'
        self.p2sButton['text'] = f'Spider {player2Hand.s}'
        self.p2bButton['text'] = f'Beetle {player2Hand.b}'
        self.p2qButton['text'] = f'Queen {player2Hand.q}'


class Hex():
    def __init__(self, id, x, y, canvas, canvasX, canvasY):
        self.id = id
        self.x = x
        self.y = y
        self.canvas = canvas
        self.canvasX = canvasX
        self.canvasY = canvasY
        self.text = None
        self.piece = None

    def highlight(self):
        self.canvas.itemconfig(self.id, outline='blue')
        self.canvas.tag_raise(self.id)

    def clearHighlight(self):
        self.canvas.itemconfig(self.id, outline='black')

    def setPiece(self, piece, stackSize):
        text = piece
        if stackSize != 1:
            text = f'{stackSize} {piece}'
        self.text = self.canvas.create_text(
            self.canvasX, self.canvasY, text=text)
        self.piece = piece


class Hexes():
    def __init__(self):
        self.hexes = []
        self.highlighted = []

    def add(self, hex):
        self.hexes.append(hex)

    def getById(self, id):
        return next((hex for hex in self.hexes if hex.id == id or hex.text == id), None)

    def getByXY(self, x, y):
        return next((hex for hex in self.hexes if (hex.x == x and hex.y == y)), None)

    def highlight(self, x, y):
        hex = self.getByXY(x, y)
        hex.highlight()
        self.highlighted.append(hex)

    def clearHighlighted(self):
        for hex in self.highlighted:
            hex.clearHighlight()
        self.highlighted = []


class GUI():
    def __init__(self):

        self.hexes = Hexes()

        self.piecePlaceMode = False
        self.pieceToPlace = None

        self.moveMode = False
        self.hexToMove = None

        window = self.setupWindow()

        self.state = hive.newGame()
        self.handDisplay.updateHandDisplay(
            self.state.player1Hand, self.state.player2Hand)
        self.infoDisplay.update(
            self.state.player, self.state.turn, self.state.terminal, self.state.terminalValue)
        self.drawBoard()

        window.mainloop()

    def drawBoard(self):
        size = 20
        yoff = 50
        xoff = 50
        h = 2*size
        w = math.sqrt(3)*size
        for j in range(23):
            x = xoff + j*w/2
            y = j*3*h/4 + yoff
            for i in range(23):
                id = self.drawHex(x, y, size)
                hex = Hex(id, i, j, self.canvas, x, y)
                stackSize, piece = stackSizeAndTopPiece(
                    i, j, self.state.board)
                if stackSize != 0:
                    hex.setPiece(piece, stackSize)
                self.hexes.add(hex)
                x += w

    def drawHex(self, x, y, size):
        points = []
        for i in range(6):
            angle = math.pi/180*(60*i-30)
            points.append(x+size*math.cos(angle))
            points.append(y+size*math.sin(angle))
        return self.canvas.create_polygon(
            points, width=3, fill="", outline='black')

    def onCanvasClick(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        hexId = event.widget.find_overlapping(x, y, x+1, y+1)
        if(len(hexId) == 0):
            return
        self.onHexClick(hexId[0])

    def onHexClick(self, hexId):
        hex = self.hexes.getById(hexId)
        if self.piecePlaceMode:
            self.tryPlacePiece(hex)
        elif self.moveMode:
            self.tryMovePiece(hex)
        elif hex.piece:
            self.startMoveAction(hex)

    def startMoveAction(self, hex):
        moveActions = self.state.availableActions.getMoveActionsByStart(
            hex.x, hex.y)
        if len(moveActions) == 0:
            return
        self.moveMode = True
        self.hexToMove = hex
        for action in moveActions:
            self.hexes.highlight(action.endX, action.endY)

    def onSelectHandPiece(self, piece):

        self.piecePlaceMode = False
        self.moveMode = False

        self.hexes.clearHighlighted()

        if not self.state.availableActions.canBePlaced(piece):
            return
        self.piecePlaceMode = True
        self.pieceToPlace = piece
        for spot in self.state.availableActions.placeAction.spots:
            self.hexes.highlight(spot[0], spot[1])

    def tryPlacePiece(self, hex):
        self.clearModes()
        if not self.state.availableActions.canBePlacedAt(self.pieceToPlace, hex.x, hex.y):
            return
        self.doAction(self.state.availableActions.placeAction,
                      self.pieceToPlace, hex.x, hex.y)

    def tryMovePiece(self, hex):
        self.clearModes()
        action = self.state.availableActions.getMoveAction(
            self.hexToMove.x, self.hexToMove.y, hex.x, hex.y)
        if action is not None:
            self.doAction(action)

    def clearModes(self):
        self.moveMode = False
        self.piecePlaceMode = False
        self.hexes.clearHighlighted()

    def doAction(self, action, *actionArgs):
        self.state = action.do(self.state, *actionArgs)
        self.canvas.delete("all")
        self.hexes = Hexes()
        self.drawBoard()
        self.handDisplay.updateHandDisplay(
            self.state.player1Hand, self.state.player2Hand)
        self.infoDisplay.update(
            self.state.player, self.state.turn, self.state.terminal, self.state.terminalValue)

    def setupWindow(self):
        window = tk.Tk()
        window.resizable(width=False, height=False)

        frame = tk.Frame(window)
        frame.pack()
        self.infoDisplay = InfoDisplay(frame)
        self.handDisplay = HandDisplay(
            frame, self.onSelectHandPiece)

        self.setupBoardCanvas(frame)
        return window

    def setupBoardCanvas(self, frame):
        self.canvas = tk.Canvas(frame, width=500, height=500,
                                scrollregion=(0, 0, 1220, 750), bg='white')

        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.yview_moveto(0.17)
        self.canvas.xview_moveto(0.30)  # center - excuse the magic numbers
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.onCanvasClick)


GUI()
