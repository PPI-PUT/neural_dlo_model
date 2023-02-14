import tensorflow as tf

from utils.constants import BSplineConstants


class SeparatedCNNNeuralPredictor(tf.keras.Model):
    def __init__(self):
        super(SeparatedCNNNeuralPredictor, self).__init__()
        activation = tf.keras.activations.tanh
        N = 128
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
            tf.keras.layers.Conv1D(8, 3, activation=activation),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(16, 3, activation=activation),
            tf.keras.layers.MaxPool1D(),
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            #tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(N, activation),
        ]
        self.fc = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(32*4, activation),
            #tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(BSplineConstants.ncp),
        ]
        self.cnn = [
            tf.keras.layers.Conv1DTranspose(32, 3, activation=activation),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Conv1DTranspose(16, 3, padding='same', activation=activation),
            tf.keras.layers.UpSampling1D(),
            tf.keras.layers.Conv1DTranspose(8, 3, activation=activation),
            tf.keras.layers.Conv1DTranspose(3, 1),
        ]

    def __call__(self, rot, trans, cable, training=False):
        for l in self.rot:
            rot = l(rot, training=training)
        for l in self.trans:
            trans = l(trans, training=training)
        cable = tf.reshape(cable, (-1, BSplineConstants.n, BSplineConstants.dim))
        for l in self.cable:
            cable = l(cable, training=training)

        x = tf.concat([rot, trans, cable], axis=-1)
        for l in self.fc:
            x = l(x, training=training)
        x = tf.reshape(x, (-1, 4, 32))
        for l in self.cnn:
            x = l(x, training=training)
        x = x[:, :BSplineConstants.n]
        x = tf.reshape(x, (-1, BSplineConstants.n * BSplineConstants.dim))
        return x