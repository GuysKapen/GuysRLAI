import tensorflow as tf
import tensorflow.keras.layers as layers


class ScaledDotProductAttention(tf.keras.Model):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        q, k, v = inputs
        attn = tf.matmul(q / self.temperature, tf.transpose(k, perm=(0, 1, 3, 2)))
        if mask is not None:
            attn = tf.where(mask == 0, -1e9, attn)
        attn = self.dropout(attn)
        output = tf.matmul(attn, v)

        return output, attn

    def get_config(self):
        return {'temperature': self.temperature, 'dropout': self.dropout}


class MultiHeadAttention(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, n_head, d_model, d_k, d_v, dropout=.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = layers.Dense(n_head * d_k, use_bias=False)
        self.w_k = layers.Dense(n_head * d_k, use_bias=False)
        self.w_v = layers.Dense(n_head * d_v, use_bias=False)

        self.fc = layers.Dense(d_model, use_bias=False)

        self.attn = ScaledDotProductAttention(temperature=d_k ** 0.25)

        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None, mask=None):
        q, k, v = inputs
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = q
        q = self.norm(q)

        # Pass through the pre-attention projection: b x len_q x (n * d_v)
        # Separate heads: b x len_q x n x d_v

        q = tf.reshape(self.w_q(q), shape=(batch_size, len_q, n_head, d_k))
        k = tf.reshape(self.w_k(k), shape=(batch_size, len_k, n_head, d_k))
        v = tf.reshape(self.w_v(v), shape=(batch_size, len_v, n_head, d_v))

        # Transpose for attention dot product
        q, k, v = tf.transpose(q, perm=(0, 2, 1, 3)), tf.transpose(k, perm=(0, 2, 1, 3)), tf.transpose(v, perm=(
        0, 2, 1, 3))

        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)  # broadcasting

        q, attn = self.attn((q, k, v), mask=mask)

        # q = (b, n, l_q, d_k), k = (b, n, l_k, d_k), v = (b, n, l_v, d_v), attn = q x k_t = (b, n, l_q, l_k)
        # v = (b, n, l_v, d_v), l_v = l_k
        # attn x v = (b, n, l_q, d_v)

        # Transpose back b x l_q x n x d_v and merge heads b x l_q x (n*d_v)
        q = tf.reshape(tf.transpose(q, perm=(0, 2, 1, 3)), shape=(batch_size, len_q, -1))
        q = self.dropout(self.fc(q))

        # q = (b, l_q, n * d_v) x (n * d_v, d_m) = (b, l_q, d_m)
        q = q + residual

        return q, attn


class PositionWiseFeedForward(tf.keras.Model):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = layers.Dense(d_hid)
        self.w_2 = layers.Dense(d_in)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=None, mask=None):
        residual = inputs
        x = self.norm(inputs)
        x = self.w_2(tf.nn.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + inputs

        return x

    def get_config(self):
        pass
