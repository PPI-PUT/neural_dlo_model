from copy import copy

import tensorflow as tf

from utils.constants import BSplineConstants


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        activation = tf.keras.activations.tanh
        N = 256
        p = 0.2

        self.trans = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])
        self.rot = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

        self.transrot = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

        self.cable_down = [
            tf.keras.layers.Conv1D(16, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(32, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(128, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(256, 3, padding='same', activation=activation),
        ]

        self.cable_up = [
            tf.keras.layers.Conv1DTranspose(256, 3, padding='same', activation=activation),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Conv1DTranspose(128, 3, padding='same', activation=activation),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Conv1DTranspose(64, 3, padding='same', activation=activation),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Conv1DTranspose(32, 3, padding='same', activation=activation),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Conv1DTranspose(16, 3, padding='same', activation=activation),
            tf.keras.layers.Conv1DTranspose(3, 1, padding='same'),
        ]

    def __call__(self, rotation, translation, cable, training=False):
        cable_ = copy(cable)
        #R_l_0, R_l_1, R_r_0, R_r_1 = rotation
        #t_l_0, t_l_1 = translation

        rot = tf.concat(rotation, axis=-1)
        trans = tf.concat(translation, axis=-1)

        rot = self.rot(rot, training=training)
        trans = self.trans(trans, training=training)
        transrot = tf.concat([rot, trans], axis=-1)
        transrot = self.transrot(transrot, training=training)
        #transrot = tf.tile(transrot[:, tf.newaxis], (1, BSplineConstants.n, 1))

        intermediate = []
        #dcable = cable[:, 1:] - cable[:, :-1]
        #cable = tf.concat([dcable, tf.zeros_like(dcable[:, :1])], axis=1)
        for l in self.cable_down:
            cable = l(cable, training=training)
            intermediate.append(copy(cable))

        #x = tf.concat([transrot[:, tf.newaxis], cable], axis=-1)
        x = (transrot[:, tf.newaxis] + cable) / 2
        #x = transrot[:, tf.newaxis] + cable

        for i, l in enumerate(self.cable_up):
            x = l(x, training=training)
            #if l.name.startswith("up_sampling"):
            #    if len(intermediate)-i-2 > 0:
            #        x = tf.concat([x, intermediate[-i-2]], axis=-1)
        x = x + cable_
        return x