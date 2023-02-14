import tensorflow as tf

from utils.constants import BSplineConstants


class BasicNeuralPredictor(tf.keras.Model):
    def __init__(self):
        super(BasicNeuralPredictor, self).__init__()

        activation = tf.keras.activations.tanh
        N = 256
        self.fc = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp),
        ]

    def __call__(self, x):
        for l in self.fc:
            x = l(x)
        return x