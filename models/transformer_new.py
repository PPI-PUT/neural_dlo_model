import tensorflow as tf
import numpy as np
from keras.layers import Dense

from utils.constants import BSplineConstants


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
        self.embedding = Dense(d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff):
        super(DecoderLayer, self).__init__()

        self.global_self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.global_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff)
            for _ in range(num_layers)]


        self.last_attn_scores = None

    def call(self, x, context):
        # `x` is token-IDs shape (batch, target_seq_len)
        x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)


        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores

        # The shape of x is (batch_size, target_seq_len, d_model).
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.enc_layers = [Dense(d_model) for _ in range(num_layers)]

    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
            x = tf.keras.activations.relu(x)
            #x = tf.keras.activations.tanh(x)
        return x


class TransformerNew(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 target_size):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff)

        self.final_layer = tf.keras.layers.Dense(target_size)

    def call(self, rotation, translation, cable, true_rotation, true_translation, training=False):
        rot = tf.concat(rotation, axis=-1)
        trans = tf.concat(translation, axis=-1)
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context = tf.concat([rot, trans], axis=-1)
        x = cable

        context = self.encoder(context)[:, :, tf.newaxis]

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        x = self.final_layer(x)
        return x
