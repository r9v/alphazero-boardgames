import argparse
from .TicTacToeNNet import TicTacToeNNet as onnet
import os
import numpy as np
import sys
sys.path.append('..')


class NNetWrapper():
    def __init__(self):
        self.nnet = onnet()

    def predict(self, board):
        board = board[np.newaxis, :, :]
        pi, v = self.nnet.model.predict(board)
        #print(f'pi {pi}')
        return pi[0][:-1], v[0]

    def load_checkpoint(self):
      # C:\Users\www\Projects\hive-tensorflow\TTTNet\w.pth.tar
        filepath = os.path.abspath(
            # C:\Users\www\Downloads\alpha-zero-general-master\alpha-zero-general-master\pretrained_models\tictactoe\keras\best-25eps-25sim-10epch.pth.tar
            'C:\\Users\www\Downloads\\alpha-zero-general-master\\alpha-zero-general-master\pretrained_models\\tictactoe\keras\\best-25eps-25sim-10epch.pth.tar')

        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.nnet.model.load_weights(filepath)
