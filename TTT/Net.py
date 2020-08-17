import tensorflow as tf
import random
import numpy as np


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
