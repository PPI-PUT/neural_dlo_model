import tensorflow as tf
from keras import Sequential
from keras.layers import Bidirectional, LSTM

from utils.constants import BSplineConstants


class INBiLSTM(tf.keras.Model):
    def __init__(self):
        super(INBiLSTM, self).__init__()
        # activation = tf.keras.activations.relu
        activation = tf.keras.activations.tanh
        N = 128

        self.edge_processor = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ]

        self.point_encoder = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ])

        self.point_predictor = Sequential([
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(3, activation),
        ])

        self.bilstm = Sequential([
            # Bidirectional(LSTM(N, return_sequences=True), merge_mode="sum"),
            # Bidirectional(LSTM(N, return_sequences=True), merge_mode="sum"),
            Bidirectional(LSTM(int(N / 2), return_sequences=True)),
            Bidirectional(LSTM(int(N / 2), return_sequences=True)),
        ])

        self.action_left = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ]

        self.action_right = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ]

    def __call__(self, rotation, translation, cable, true_rotation, true_translation, training=False):
        #left_action = tf.concat([rotation[:, :18], translation], axis=-1)
        #right_action = rotation[:, 18:]
        left_action = tf.concat([rotation[0], rotation[1], *translation], axis=-1)
        right_action = tf.concat(rotation[2:], axis=-1)

        points = cable
        edges = points[:, 1:] - points[:, :-1]
        reverse_edges = points[:, :-1] - points[:, 1:]

        for l in self.edge_processor:
            edges = l(edges, training=training)

        for l in self.edge_processor:
            reverse_edges = l(reverse_edges, training=training)

        for l in self.action_left:
            left_action = l(left_action, training=training)

        for l in self.action_right:
            right_action = l(right_action, training=training)

        relations_impacts = tf.concat([reverse_edges[:, :1] + right_action[:, tf.newaxis],
                                       reverse_edges[:, 1:] + edges[:, :-1],
                                       edges[:, -1:] + left_action[:, tf.newaxis]], axis=1)

        relations_impacts = self.bilstm(relations_impacts)

        points_embedding = self.point_encoder(points, training=training)
        dpoints = tf.concat([points_embedding, relations_impacts], axis=-1)

        dpoints = self.point_predictor(dpoints, training=training)

        return dpoints
