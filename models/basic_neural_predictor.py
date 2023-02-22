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

    def __call__(self, x1, x2, x3, training=True):
        bs = x3.shape[0]
        x1 = tf.concat([tf.reshape(x, (bs, -1)) for x in x1], axis=-1)
        x2 = tf.concat([tf.reshape(x, (bs, -1)) for x in x2], axis=-1)
        x3_ = tf.reshape(x3, (bs, -1))
        x = tf.concat([x1, x2, x3_], axis=-1)
        for l in self.fc:
            x = l(x)
        x = tf.reshape(x, (-1, BSplineConstants.n, BSplineConstants.dim)) + x3
        return x