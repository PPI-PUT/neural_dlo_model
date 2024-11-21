import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow_graphics.geometry.transformation import euler

from utils.constants import BSplineConstants
import numpy as np


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
        elif rotation == "euler":
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
        #rot = tf.concat(rotation, axis=-1)
        #trans = tf.concat(translation, axis=-1)
        rot = tf.concat([rotation[0], rotation[2]], axis=-1)
        trans = translation[0]
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


class JacobianRBFN(tf.keras.Model):
    def __init__(self, rotation, diff, features):
        super(JacobianRBFN, self).__init__()
        activation = tf.keras.activations.tanh
        N = 256
        p = 0.2

        #kmeans = KMeans(n_clusters=N, n_init=2, max_iter=100).fit(features)

        #nSamples = np.zeros((N,), dtype='float32')
        #variance = np.zeros((N,), dtype='float32')
        #for i, label in enumerate(kmeans.labels_):
        #    variance[label] += np.linalg.norm(features[i, :] - kmeans.cluster_centers_[label, :]) ** 2
        #    nSamples[label] += 1
        #variance = variance / (nSamples - 1)

        #self.mu = tf.Variable(kmeans.cluster_centers_[np.newaxis], trainable=True)
        #self.var = tf.Variable(variance[np.newaxis], trainable=True)
        self.mu = tf.Variable(np.random.random((1, N, 203)).astype(np.float32), trainable=True)
        self.var = tf.Variable(np.random.random((1, N)).astype(np.float32), trainable=True)

        trans_dim = 3
        rot_dim = 0
        if rotation == "quat":
            #rot_dim = 4
            rot_dim = 3
        elif rotation == "rotvec":
            rot_dim = 3
        elif rotation == "euler":
            rot_dim = 3
        elif rotation == "rotmat":
            rot_dim = 9

        self.rotation = rotation
        self.diff = diff
        self.action_dim = trans_dim + 2 * rot_dim
        self.fc = tf.keras.layers.Dense(BSplineConstants.ncp * self.action_dim, use_bias=False)
        self.flat = tf.keras.layers.Flatten()

    def __call__(self, rotation, translation, cable, true_rotation, true_translation, training=False):
        #rot = tf.concat(rotation, axis=-1)
        #trans = tf.concat(translation, axis=-1)
        rot = tf.concat([rotation[0], rotation[2]], axis=-1)
        trans = translation[0]
        cable = self.flat(cable)
        x = tf.concat([rot, trans, cable], axis=-1)

        cluster_dists = tf.reduce_sum(tf.square(x[:, tf.newaxis] - self.mu), axis=-1)
        exponent = -cluster_dists / self.var
        thetas = np.exp(exponent)

        J = self.fc(thetas)
        J = tf.reshape(J, (-1, BSplineConstants.n, BSplineConstants.dim, self.action_dim))

        if self.diff:
            rl = euler.from_quaternion(true_rotation[1])
            rr = euler.from_quaternion(true_rotation[3])
            action = tf.concat([rl, rr, true_translation[1]], axis=-1)
            #action = tf.concat([true_rotation[1], true_rotation[3], true_translation[1]], axis=-1)
        else:
            rl = euler.from_quaternion(true_rotation[1]) - euler.from_quaternion(true_rotation[0])
            rr = euler.from_quaternion(true_rotation[3]) - euler.from_quaternion(true_rotation[2])
            action = tf.concat([rl, rr, true_translation[1] - true_translation[0]], axis=-1)
            #action = tf.concat([true_rotation[1] - true_rotation[0],
            #                    true_rotation[3] - true_rotation[2],
            #                    true_translation[1] - true_translation[0]], axis=-1)
        x = (J @ action[:, tf.newaxis, :, tf.newaxis])[..., 0]
        return x
