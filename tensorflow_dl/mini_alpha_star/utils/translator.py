import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.libs.models import get_subsequent_mask, get_pad_mask


class Translator(tf.keras.Model):
    def __init__(self, model, beam_size, max_seq_len,
                 src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        super(Translator, self).__init__()
        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model

        self.init_seq = tf.convert_to_tensor([[trg_bos_idx]], dtype=tf.int64)
        self.blank_seqs = tf.convert_to_tensor(tf.fill((beam_size, max_seq_len), trg_pad_idx), dtype=tf.int64)
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.len_map = tf.expand_dims(tf.range(1, max_seq_len + 1, dtype=tf.int64), axis=0)

    def _model_decode(self, trg_seq, enc_output, src_mask):
        trg_mask = get_subsequent_mask(trg_seq)
        dec_output, *_ = self.model.decoder((trg_seq, enc_output), mask=(trg_mask, src_mask))
        return tf.math.softmax(self.model.trg_word_proj(dec_output), axis=-1)

    def _get_init_state(self, src_seq, src_mask):
        beam_size = self.beam_size
        enc_output, *_ = self.model.encoder(src_seq, mask=src_mask)
        dec_output = self._model_decode(self.init_seq, enc_output, src_mask)

        best_k_probs, best_k_idx = tf.math.top_k(dec_output[:, -1, :], beam_size)

        scores = tf.reshape(tf.math.log(best_k_probs), beam_size)
        gen_seq = tf.convert_to_tensor(self.blank_seqs)
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = tf.tile(enc_output, tf.constant([beam_size, 1, 1]))

        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.shape) == 1

        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total
        best_k2_probs, best_k2_idx = tf.math.top_k(dec_output[:, -1, :], beam_size)

        # Include the previous scores
        scores = tf.reshape(tf.math.log(best_k2_probs), shape=(beam_size, -1)) + tf.reshape(scores,
                                                                                            shape=(beam_size, 1))
        # Get the best k candidates from k^2 candidates
        scores, best_k_idx_in_k2 = tf.math.top_k(tf.reshape(scores, shape=-1), beam_size)

        # Get the corresponding positions of the best k candidates
        best_k_r_idxes, best_k_c_idxes = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxes, best_k_c_idxes]

        # Copy the corresponding previous tokens
        gen_seq[:, :step] = gen_seq[best_k_r_idxes, :step]
        # SEt the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores

    def translate_sentence(self, src_seq):
        """

        :param src_seq:
        :return:
        """
        # Only accept batch size equals 1
        assert src_seq.shape[0] == 1

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        src_mask = get_pad_mask(src_seq, src_pad_idx)
        enc_output, gen_seq, scores = self._get_init_state(src_seq, src_mask)

        ans_idx = 0
        # Decode up to max length
        for step in range(2, max_seq_len):
            dec_output = self._model_decode(gen_seq[:, :step], enc_output, src_mask)
            gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

            # Check if all path finished
            # Locate the eos in the generated sequences
            eos_locs = gen_seq == trg_eos_idx
            # Replace the eos with its position for the length penalty use
            seq_lens = tf.reduce_min(tf.where(~eos_locs, max_seq_len, self.len_map), axis=1)
            # Check if all beams contain eos
            if tf.reduce_sum((tf.reduce_sum(eos_locs, axis=1) > 0), axis=0) == beam_size:
                ans_idx = tf.argmax(scores / tf.reduce_max((tf.cast(seq_lens, dtype=tf.float32) ** alpha)), axis=0)
                break


        return list(gen_seq[ans_idx][:seq_lens[ans_idx]])


