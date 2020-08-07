import tkinter as tk
import math
from Hive import Hive

hive = Hive()


class GUI():
    def __init__(self):
        window = self.setupWindow()
        board, player1Hand, player2Hand, currentPlayer = hive.newGame()
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

        b1 = tk.Button(frame, width=10)
        b2 = tk.Button(frame, width=10)
        b1.pack(side=tk.LEFT)
        b2.pack(side=tk.RIGHT)

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


GUI()
