import tensorflow as tf
import random
import numpy as np
import os
import sys
import time


class Net():
    def __init__(self):
        self.model = self.buildModel()

    def convLayer(self, x):
        x = tf.keras.layers.Conv2D(data_format='channels_first',
                                   filters=256, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(
                                       1e-4)
                                   )(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def resLayer(self, xIn):
        x = tf.keras.layers.Conv2D(data_format='channels_first',
                                   filters=256, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(
                                       1e-4),
                                   )(xIn)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(data_format='channels_first',
                                   filters=256, kernel_size=(3, 3), padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(
                                       1e-4)
                                   )(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.Add()([xIn, x])
        x = tf.keras.layers.ReLU()(x)
        return x

    def valueHead(self, x):
        x = tf.keras.layers.Conv2D(data_format='channels_first',
                                   filters=1, kernel_size=(1, 1),
                                   padding='same', activation='linear',
                                   kernel_regularizer=tf.keras.regularizers.l2(
                                       1e-4)
                                   )(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='linear')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(1, activation='tanh')(x)
        return x

    def policyHead(self, x):
        x = tf.keras.layers.Conv2D(data_format='channels_first',
                                   filters=2, kernel_size=(1, 1),
                                   padding='same', activation='linear',
                                   kernel_regularizer=tf.keras.regularizers.l2(
                                       1e-4)
                                   )(x)
        x = tf.keras.layers.BatchNormalization(axis=1)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(9, activation='softmax')(x)

        return x

    def buildModel(self):
        theInput = tf.keras.Input(shape=(8, 3, 3))  # 8 channels of 3x3 boards

        x = self.convLayer(theInput)

        for i in range(2):
            x = self.resLayer(x)

        valueHead = self.valueHead(x)
        policyHead = self.policyHead(x)

        model = tf.keras.Model(inputs=theInput, outputs=[
                               valueHead, policyHead])

        model.compile(loss=['mean_squared_error',
                            'categorical_crossentropy'], optimizer=tf.keras.optimizers.Adam(0.001))
        return model

    def predict(self, state):
        theInput = self.stateToInput(state)
        v, pi = self.model.predict(np.array([theInput]))
        return v[0][0], pi[0]

    def stateToInput(self, state):
        theInput = np.zeros((8, 3, 3), dtype=int)

        if(state.prevState):
            oneHot = self.boardToOneHot(state.prevState.board)
            theInput[2] = oneHot[0]
            theInput[3] = oneHot[1]
            if(state.prevState.prevState):
                oneHot = self.boardToOneHot(state.prevState.prevState.board)
                theInput[0] = oneHot[0]
                theInput[1] = oneHot[1]

        oneHot = self.boardToOneHot(state.board)
        theInput[4] = oneHot[0]
        theInput[5] = oneHot[1]
        if(state.player == -1):
            theInput[6][:] = 1
        else:
            theInput[7][:] = 1
        return theInput

    def boardToOneHot(self, board):
        oneHot = np.zeros((2, 3, 3), dtype=int)
        firstPlayer = np.where(board == -1)
        oneHot[0][firstPlayer] = 1
        secondPlayer = np.where(board == 1)
        oneHot[1][secondPlayer] = 1
        return oneHot

    def save(self):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filePath = os.path.join('TTT', 'networks', f'{timestr}.h5')
        self.model.save_weights(filePath, overwrite=False)

        ckpt = self.getLatestModelFileName()

        f = open(os.path.join('TTT', 'networks', 'latest.txt'), "w")
        f.write(f'{timestr}.h5')
        f.close()

        f = open(os.path.join('TTT', 'networks', 'secondLatest.txt'), "w")
        f.write(ckpt)
        f.close()
        print(f'Model {timestr}.h5  saved')
        return f'{timestr}.h5'

    def loadLatest(self):
        ckpt = self.getLatestModelFileName()
        if self.load(ckpt):
            print('Loaded latest checkpoint')
        else:
            print('Failed to load latest checkpoint')

    def loadSecondLatest(self):
        f = open(os.path.join('TTT', 'networks', 'secondLatest.txt'), "r")
        ckpt = f.read()
        f.close()
        if self.load(ckpt):
            print('Loaded second latest checkpoint')
        else:
            print('Failed to load second latest checkpoint')

    def load(self, name):
        filePath = os.path.join('TTT', 'networks', name)
        if name == '' or not os.path.exists(filePath):
            print(f'No checkpoint named {name}')
            return False
        self.model.load_weights(filePath)
        print(f'Model loaded from {name} checkpoint')
        return True

    def getLatestModelFileName(self):
        f = open(os.path.join('TTT', 'networks', 'latest.txt'), "r")
        ckpt = f.read()
        f.close()
        return ckpt
