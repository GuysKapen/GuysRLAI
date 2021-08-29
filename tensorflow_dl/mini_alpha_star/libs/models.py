import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from tensorflow_dl.mini_alpha_star.libs.transformer import EncoderLayer, DecoderLayer


def get_pad_mask(seq, pad_idx):
    return tf.expand_dims((seq != pad_idx), axis=-2)


def get_subsequent_mask(seq):
    """
    Masking out the subsequent info
    :param seq:
    :return:
    """
    b_s, len_s = seq.shape
    subsequent_mask = tf.cast(
        (1 - tf.linalg.band_part(tf.ones(shape=(1, len_s, len_s)) - tf.ones(shape=(1, len_s, len_s)), 0, -1)),
        dtype=tf.bool)
    return subsequent_mask


class PositionalEncoding(tf.keras.Model):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.pos_table = self._get_sinusoid_encoding_table(n_position, d_hid)

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_hid):
        """
        Sinusoid position encoding table
        :param n_position:
        :param d_hid:
        :return:
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 2i + 1

        return tf.expand_dims(tf.convert_to_tensor(sinusoid_table, dtype=tf.int64), axis=0)

    def call(self, x, training=None, mask=None):
        return tf.stop_gradient(x + self.pos_table[:, :x.shape[1]])

    def get_config(self):
        pass


class Encoder(tf.keras.Model):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model,
                 d_inner, pad_idx, dropout=0.1, n_position=200):
        super(Encoder, self).__init__()
        self.src_word_emb = layers.Embedding(n_src_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = layers.Dropout(dropout)
        self.layer_stack = [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, src_seq, training=None, mask=None, return_attns=False):
        enc_slf_attn_list = []

        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

    def get_config(self):
        pass


class Decoder(tf.keras.Model):
    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_head,
                 d_k, d_v, d_model, d_inner, pad_idx, n_position=200, dropout=0.1):
        super(Decoder, self).__init__()

        self.trg_word_emb = layers.Embedding(n_trg_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = layers.Dropout(dropout)
        self.layer_stack = [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout) for _ in range(n_layers)]
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None, mask=None, return_attns=False):
        trg_seq, enc_output = inputs
        trg_mask, src_mask = mask
        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer((dec_output, enc_output), mask=(trg_mask, src_mask))
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        dec_output = self.layer_norm(dec_output)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

    def get_config(self):
        pass


class Transformer(tf.keras.Model):
    def __init__(self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
                 n_position=200, trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True):
        super(Transformer, self).__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout
        )

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, d_word_vec=d_word_vec,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx,
            n_position=n_position, dropout=dropout
        )

        self.trg_word_proj = layers.Dense(n_trg_vocab, use_bias=False)

        assert d_model == d_word_vec, 'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_proj.set_weights(self.decoder.trg_word_emb.get_weights())
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.set_weights(self.decoder.trg_word_emb.get_weights())

    def call(self, inputs, training=None, mask=None):
        src_seq, trg_seq = inputs
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, mask=src_mask)
        dec_output, *_ = self.decoder((trg_seq, enc_output), mask=(trg_mask, src_mask))
        seq_logit = self.trg_word_proj(dec_output) * self.x_logit_scale

        return tf.reshape(seq_logit, shape=(-1, seq_logit.shape[2]))
