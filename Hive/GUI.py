import tkinter as tk
import math
from Hive import Hive
from const import *

hive = Hive()


class GUI():
    def __init__(self):
        window = self.setupWindow()
        board, player1Hand, player2Hand, currentPlayer = hive.newGame()
        self.handDisplay.updateHandDisplay(player1Hand, player2Hand)
        availableActions = hive.availableActions(
            board, player1Hand, currentPlayer)
        print(availableActions)
        self.drawBoard(board)
        window.mainloop()

    def drawBoard(self, board):
        size = 20
        yoff = 50
        xoff = 50
        w = 2*size
        h = math.sqrt(3)*size
        self.hexes = {}
        for j in range(23):
            x = xoff
            y = j*h + yoff
            for i in range(23):
                color = 'white'
                if i == 11 and j == 11:
                    color = 'green'
                id = self.drawHex(x, y, size, color)
                self.hexes[id] = (i, j)
                x += 3*w/4
                y += h/2

    def drawHex(self, x, y, size, color):
        points = []
        for i in range(6):
            angle = math.pi/180*(60*i)
            points.append(x+size*math.cos(angle))
            points.append(y+size*math.sin(angle))

        return self.canvas.create_polygon(
            points, width=3, fill=color, outline='black')

    def click(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        hexId = event.widget.find_overlapping(x, y, x+1, y+1)
        if(len(hexId) == 0):
            return
        hexId = hexId[0]
        print(self.hexes[hexId])

    def setupWindow(self):
        window = tk.Tk()
        window.resizable(width=False, height=False)

        frame = tk.Frame(window)
        frame.pack()

        self.handDisplay = HandDisplay(frame)
        self.setupBoardCanvas(frame)
        return window

    def setupBoardCanvas(self, frame):
        self.canvas = tk.Canvas(frame, width=500, height=500,
                                scrollregion=(0, 0, 750, 1220), bg='Black')

        hbar = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        vbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.yview_moveto(0.30)
        self.canvas.xview_moveto(0.17)  # center - excuse the magic numbers
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.click)


class HandDisplay():
    def __init__(self, frame):
        leftFrame = tk.Frame(frame)
        rightFrame = tk.Frame(frame)
        leftFrame.pack(side=tk.LEFT)
        rightFrame.pack(side=tk.RIGHT)

        self.p1aButton = tk.Button(
            leftFrame, width=10, command=lambda: self.action(Player1A))
        self.p1gButton = tk.Button(
            leftFrame, width=10, command=lambda: self.action(Player1G))
        self.p1sButton = tk.Button(
            leftFrame, width=10, command=lambda: self.action(Player1S))
        self.p1bButton = tk.Button(
            leftFrame, width=10, command=lambda: self.action(Player1B))
        self.p1qButton = tk.Button(
            leftFrame, width=10, command=lambda: self.action(Player1Q))

        self.p2aButton = tk.Button(
            rightFrame, width=10, command=lambda: self.action(Player2A))
        self.p2gButton = tk.Button(
            rightFrame, width=10, command=lambda: self.action(Player2G))
        self.p2sButton = tk.Button(
            rightFrame, width=10, command=lambda: self.action(Player2S))
        self.p2bButton = tk.Button(
            rightFrame, width=10, command=lambda: self.action(Player2B))
        self.p2qButton = tk.Button(
            rightFrame, width=10, command=lambda: self.action(Player2Q))

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

    def action(self, number):
        print(number)


GUI()
