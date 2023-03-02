import tensorflow as tf

from utils.constants import BSplineConstants


class ScaleNeuralPredictor(tf.keras.Model):
    def __init__(self):
        super(ScaleNeuralPredictor, self).__init__()
        activation = tf.keras.activations.tanh
        N = 256
        p = 0.2

        self.cable = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

        self.rot_l = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp * BSplineConstants.dim, activation),
            tf.keras.layers.Reshape((BSplineConstants.n, BSplineConstants.dim, BSplineConstants.dim))
        ])

        self.rot_r = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp * BSplineConstants.dim, activation),
            tf.keras.layers.Reshape((BSplineConstants.n, BSplineConstants.dim, BSplineConstants.dim))
        ])

        self.trans_l = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp * BSplineConstants.dim, activation),
            tf.keras.layers.Reshape((BSplineConstants.n, BSplineConstants.dim, BSplineConstants.dim))
        ])


    def __call__(self, rotation, translation, cable, training=False):
        left_arm_0 = translation[0]
        trans_l = translation[1]
        rot_l = rotation[1]
        rot_r = rotation[3]

        cable_state = tf.concat([cable, left_arm_0[:, tf.newaxis]], axis=1)

        cable_state = self.cable(cable_state, training=training)

        cable_rot_l = self.rot_l(cable_state, training=training)
        cable_rot_r = self.rot_r(cable_state, training=training)
        cable_trans_l = self.trans_l(cable_state, training=training)

        dcable_rot_l = cable_rot_l @ rot_l[:, tf.newaxis, :, tf.newaxis]
        dcable_rot_r = cable_rot_r @ rot_r[:, tf.newaxis, :, tf.newaxis]
        dcable_trans_l = cable_trans_l @ trans_l[:, tf.newaxis, :, tf.newaxis]

        x = dcable_rot_l + dcable_rot_r + dcable_trans_l

        x = tf.reshape(x, (-1, BSplineConstants.n, BSplineConstants.dim))
        return x


class ScaleNeuralPredictor1(tf.keras.Model):
    def __init__(self):
        super(ScaleNeuralPredictor1, self).__init__()
        activation = tf.keras.activations.tanh
        N = 256
        p = 0.2

        self.cable = tf.keras.sequential([
            tf.keras.layers.flatten(),
            tf.keras.layers.dense(N, activation),
            tf.keras.layers.dense(N, activation),
            tf.keras.layers.dense(N, activation),
        ])

        self.pre_rot_l = tf.keras.sequential([
            tf.keras.layers.dense(N, activation),
            tf.keras.layers.dense(N, activation),
        ])

        self.pre_rot_r = tf.keras.sequential([
            tf.keras.layers.dense(N, activation),
            tf.keras.layers.dense(N, activation),
        ])


        self.pre_trans_l = tf.keras.sequential([
            tf.keras.layers.dense(N, activation),
            tf.keras.layers.dense(N, activation),
        ])

        self.rot_l = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp * BSplineConstants.dim, activation),
            tf.keras.layers.Reshape((BSplineConstants.n, BSplineConstants.dim, BSplineConstants.dim))
        ])

        self.rot_r = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp * BSplineConstants.dim, activation),
            tf.keras.layers.Reshape((BSplineConstants.n, BSplineConstants.dim, BSplineConstants.dim))
        ])

        self.trans_l = tf.keras.Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(BSplineConstants.ncp * BSplineConstants.dim, activation),
            tf.keras.layers.Reshape((BSplineConstants.n, BSplineConstants.dim, BSplineConstants.dim))
        ])


    def __call__(self, rotation, translation, cable, training=False):
        left_arm_0 = translation[0]
        trans_l = translation[1]
        rot_l = rotation[1]
        rot_r = rotation[3]

        cable_state = tf.concat([cable, left_arm_0[:, tf.newaxis]], axis=1)

        cable_state = self.cable(cable_state, training=training)

        pre_rot_l = self.pre_rot_l(rot_l)
        pre_rot_r = self.pre_rot_r(rot_r)
        pre_trans_l = self.pre_trans_l(trans_l)

        cable_state_rot_l = tf.concat([cable_state, pre_rot_l])
        cable_state_rot_r = tf.concat([cable_state, pre_rot_r])
        cable_state_trans_l = tf.concat([cable_state, pre_trans_l])

        cable_rot_l = self.rot_l(cable_state_rot_l, training=training)
        cable_rot_r = self.rot_r(cable_state_rot_r, training=training)
        cable_trans_l = self.trans_l(cable_state_trans_l, training=training)

        dcable_rot_l = cable_rot_l @ rot_l[:, tf.newaxis, :, tf.newaxis]
        dcable_rot_r = cable_rot_r @ rot_r[:, tf.newaxis, :, tf.newaxis]
        dcable_trans_l = cable_trans_l @ trans_l[:, tf.newaxis, :, tf.newaxis]

        x = dcable_rot_l + dcable_rot_r + dcable_trans_l

        x = tf.reshape(x, (-1, BSplineConstants.n, BSplineConstants.dim))
        return x
