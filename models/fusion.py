import tensorflow as tf
from keras import Sequential

from utils.constants import BSplineConstants


class Fusion(tf.keras.Model):
    def __init__(self):
        super(Fusion, self).__init__()
        activation = tf.keras.activations.tanh
        N = 128
        p = 0.2

        self.trans = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])
        self.rot = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

        self.transrot = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

        self.cable_embedding = [
            tf.keras.layers.Conv1D(8, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1D(16, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1D(32, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1D(N, 3, padding='same', activation=activation),
        ]

        self.cable_prediction = [
            tf.keras.layers.Conv1D(64, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1D(16, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1D(3, 1, padding='same'),
        ]

    def __call__(self, rot, trans, cable, training=False):
        rot = self.rot(rot, training=training)
        trans = self.trans(trans, training=training)
        transrot = tf.concat([rot, trans], axis=-1)
        transrot = self.transrot(transrot, training=training)
        transrot = tf.tile(transrot[:, tf.newaxis], (1, BSplineConstants.n, 1))

        cable = tf.reshape(cable, (-1, BSplineConstants.n, BSplineConstants.dim))
        for l in self.cable_embedding:
            cable = l(cable, training=training)

        x = tf.concat([cable, transrot], axis=-1)

        for l in self.cable_prediction:
            x = l(x, training=training)
        x = tf.reshape(x, (-1, BSplineConstants.n * BSplineConstants.dim))
        return x