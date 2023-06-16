import tensorflow as tf

from utils.constants import BSplineConstants


class JacobianNeuralPredictor(tf.keras.Model):
    def __init__(self, rotation, diff):
        super(JacobianNeuralPredictor, self).__init__()
        activation = tf.keras.activations.tanh
        N = 256
        p = 0.2

        self.trans = [
            tf.keras.layers.Dense(N, activation),
            # tf.keras.layers.Dense(N, activation),
        ]
        self.rot = [
            tf.keras.layers.Dense(N, activation),
            # tf.keras.layers.Dense(N, activation),
        ]
        self.cable = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(N, activation),
            # tf.keras.layers.Dense(N, activation),
        ]

        trans_dim = 3
        rot_dim = 0
        if rotation == "quat":
            rot_dim = 4
        elif rotation == "rotvec":
            rot_dim = 3
        elif rotation == "rotmat":
            rot_dim = 9

        self.rotation = rotation
        self.diff = diff
        self.action_dim = trans_dim + 2 * rot_dim
        self.fc = [
            # tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(N, activation),
            # tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(N, activation),
            # tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(N, activation),
            # tf.keras.layers.Dropout(p),
            tf.keras.layers.Dense(BSplineConstants.ncp * self.action_dim),
        ]

    def __call__(self, rotation, translation, cable, true_rotation, true_translation, training=False):
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
        J = tf.reshape(x, (-1, BSplineConstants.n, BSplineConstants.dim, self.action_dim))

        if self.diff:
            action = tf.concat([true_rotation[1], true_rotation[3], true_translation[1]], axis=-1)
        else:
            action = tf.concat([true_rotation[1] - true_rotation[0],
                               true_rotation[3] - true_rotation[2],
                               true_translation[1] - true_translation[0]], axis=-1)
        x = (J @ action[:, tf.newaxis, :, tf.newaxis])[..., 0]
        return x
