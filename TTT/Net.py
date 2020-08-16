import tensorflow as tf
import random
import numpy as np


class Net():
    def __init__(self):
        self.model = self.buildModel()

    def convLayer(self, x):
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def resLayer(self, xIn):
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(xIn)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([xIn, x])
        x = tf.keras.layers.ReLU()(x)
        return x

    def valueHead(self, x):
        x = tf.keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1),
            padding='same', activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(32, activation='linear')(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(1, activation='tanh')(x)
        return x

    def policyHead(self, x):
        x = tf.keras.layers.Conv2D(
            filters=2, kernel_size=(1, 1),
            padding='same', activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(9, activation='softmax')(x)

        return x

    def buildModel(self):
        theInput = tf.keras.Input(shape=(3, 3, 2))  # 2 channels of 3x3 boards

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
        return np.random.dirichlet(np.ones(9)*0.5), random.randint(-1, 1)

    def stateToInput(self, state):
        np.zeros((3, 3, 4))
