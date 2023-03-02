import tensorflow as tf

from utils.constants import BSplineConstants


class SeparatedNeuralPredictor(tf.keras.Model):
    def __init__(self):
        super(SeparatedNeuralPredictor, self).__init__()
        activation = tf.keras.activations.tanh
        N = 256
        p = 0.2

        self.trans = [
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(N, activation),
        ]
        self.rot = [
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(N, activation),
        ]
        self.cable = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(N, activation),
        ]
        self.fc = [
            #tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(BSplineConstants.ncp),
        ]

    def __call__(self, rotation, translation, cable, training=False):
        rot = tf.concat(rotation, axis=-1)
        trans = tf.concat(translation, axis=-1)
        for l in self.rot:
            rot = l(rot, training=training)
        for l in self.trans:
            trans = l(trans, training=training)

        for l in self.cable:
            cable = l(cable, training=training)

        x = tf.concat([rot, trans, cable], axis=-1)
        for l in self.fc:
            x = l(x, training=training)
        x = tf.reshape(x, (-1, BSplineConstants.n, BSplineConstants.dim))
        return x