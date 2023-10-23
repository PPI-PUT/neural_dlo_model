import tensorflow as tf
from keras import Sequential

from utils.constants import BSplineConstants


class LinearNeuralPredictor(tf.keras.Model):
    def __init__(self):
        super(LinearNeuralPredictor, self).__init__()
        activation = tf.keras.activations.tanh
        N = 64
        p = 0.2

        self.trans = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])
        self.rot = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])
        self.transrot_A1 = Sequential([
            tf.keras.layers.Dense(N*N, activation),
            tf.keras.layers.Reshape((N, N))
            #tf.keras.layers.Dense(N, activation),
        ])
        self.transrot_b1 = Sequential([
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(N, activation),
        ])
        self.transrot_A2 = Sequential([
            tf.keras.layers.Dense(BSplineConstants.ncp*N, activation),
            tf.keras.layers.Reshape((BSplineConstants.ncp, N))
            #tf.keras.layers.Dense(N, activation),
        ])
        self.transrot_b2 = Sequential([
            tf.keras.layers.Dense(BSplineConstants.ncp, activation),
            #tf.keras.layers.Dense(N, activation),
        ])
        self.cable = Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

    def __call__(self, rotation, translation, cable, true_rotation, true_translation, training=False):
        rot = tf.concat(rotation, axis=-1)
        trans = tf.concat(translation, axis=-1)
        rot = self.rot(rot)
        trans = self.trans(trans)
        cable = self.cable(cable)

        x = tf.concat([rot, trans], axis=-1)
        cable = self.transrot_A1(x) @ cable[..., tf.newaxis] + self.transrot_b1(x)[..., tf.newaxis]
        cable = self.transrot_A2(x) @ cable + self.transrot_b2(x)[..., tf.newaxis]
        x = tf.reshape(cable, (-1, BSplineConstants.n, BSplineConstants.dim))
        return x