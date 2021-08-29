import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.utils.sublayers import MultiHeadAttention, PositionWiseFeedForward


class EncoderLayer(tf.keras.Model):
    def __init__(self, d_model=256, d_inner=1024, n_head=2, d_k=128, d_v=128, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def call(self, enc_input, training=None, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn((enc_input, enc_input, enc_input), mask=self_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn

    def get_config(self):
        pass


class Encoder(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, n_layers=3, n_head=2, d_k=128, d_v=128, d_model=256, d_inner=1024, dropout=0.1):
        super(Encoder, self).__init__()
        self.dropout = layers.Dropout(dropout)
        self.stack = [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)]
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None, mask=None, return_attns=False):
        enc_self_attn_list = []

        enc_output = x
        for enc_layer in self.stack:
            enc_output, enc_self_attn = enc_layer(enc_output)
            enc_self_attn_list += [enc_self_attn] if return_attns else []

        enc_output = self.norm(enc_output)

        if return_attns:
            return enc_output, enc_self_attn_list
        return enc_output,


class DecoderLayer(tf.keras.Model):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.sef_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def call(self, inputs, training=None, mask=None):
        dec_input, enc_output = inputs
        slf_attn_mask, dec_enc_attn_mask = mask
        dec_output, dec_slf_attn = self.sef_attn((dec_input, dec_input, dec_input), mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn((dec_output, enc_output, enc_output), mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn

    def get_config(self):
        pass


class Transformer(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, d_model=512, d_inner=1024, n_layers=3, n_head=2, d_k=128, d_v=128, dropout=.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            n_layers=n_layers,
            n_head=n_head,
            d_model=d_model,
            d_inner=d_inner,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )

    def call(self, inputs, training=None, mask=None):
        enc_output, *_ = self.encoder(inputs)

        return enc_output
