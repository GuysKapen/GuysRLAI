import tensorflow as tf
import tensorflow.keras.layers as layers

from tensorflow_dl.mini_alpha_star.libs.hyper_params import Arch_Hyper_Parameters as AHP

debug = False


class Core(tf.keras.Model):
    """
    Inputs: prev_state, embedded_entity, embedded_spatial, embedded_scalar
    Outputs:
        next_state - The LSTM state for the next step
        lstm_output - The output of the LSTM
    """

    def __init__(self, embedding_dim=AHP.original_1024, hidden_dim=AHP.lstm_hidden_dim,
                 batch_size=AHP.batch_size,
                 sequence_length=AHP.sequence_length,
                 n_layers=AHP.lstm_layers, drop_prob=0.0):
        super(Core, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.lstm_layers = [
            layers.LSTM(hidden_dim, dropout=drop_prob, return_sequences=True, return_state=True)
            for _ in range(n_layers)
        ]
        # )

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # self.prev_state = None
        # self.dropout = nn.Dropout(drop_prob)
        # self.fc = nn.Linear(hidden_dim, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, embedded_scalar, embedded_entity, embedded_spatial,
                batch_size=None, sequence_length=None, hidden_state=None):
        # note: the input_shape[0] is batch_seq_size, we only transfrom it to [batch_size, seq_size, ...]
        # before input it into the lstm
        # shapes of embedded_entity, embedded_spatial, embedded_scalar are all [batch_seq_size x embedded_size]
        print('embedded_scalar.shape:', embedded_scalar.shape) if debug else None
        print('embedded_entity.shape:', embedded_entity.shape) if debug else None
        print('embedded_spatial.shape:', embedded_spatial.shape) if debug else None

        batch_seq_size = embedded_scalar.shape[0]
        print('batch_size:', batch_size) if debug else None
        print('self.batch_size:', self.batch_size) if debug else None
        batch_size = batch_size if batch_size is not None else self.batch_size
        sequence_length = sequence_length if sequence_length is not None else self.sequence_length

        print('batch_seq_size:', batch_seq_size) if debug else None
        print('batch_size:', batch_size) if debug else None
        print('sequence_length:', sequence_length) if debug else None

        assert batch_seq_size == batch_size * sequence_length
        assert batch_seq_size == embedded_entity.shape[0]
        assert batch_seq_size == embedded_spatial.shape[0]

        print('embedded_scalar is nan:', tf.reduce_any(tf.math.is_nan(embedded_scalar))) if debug else None
        print('embedded_entity is nan:', tf.reduce_any(tf.math.is_nan(embedded_entity))) if debug else None
        print('embedded_spatial is nan:', tf.reduce_any(tf.math.is_nan(embedded_spatial))) if debug else None

        input_tensor = tf.concat([embedded_scalar, embedded_entity, embedded_spatial], axis=-1)

        # note, before input to the LSTM
        # we transform the shape from [batch_seq_size, embedding_size]
        # to the actual [batch_size, seq_size, embedding_size]
        print('input_tensor.shape:', input_tensor.shape) if debug else None
        embedding_size = input_tensor.shape[-1]
        # input_tensor = input_tensor.unsqueeze(1)

        input_tensor = tf.reshape(input_tensor, shape=(batch_size, sequence_length, embedding_size))
        print('input_tensor.shape:', input_tensor.shape) if debug else None

        print('input_tensor is nan:', tf.reduce_any(tf.math.is_nan(input_tensor))) if debug else None
        
        if hidden_state is None:
            hidden_state = self.init_hidden_state(batch_size=batch_size)

        lstm_output, hidden_state = self.forward_lstm(input_tensor, hidden_state)
        # lstm_output shape: [batch_size, seq_size, hidden_dim]
        print('lstm_output.shape:', lstm_output.shape) if debug else None
        # note, after the LSTM
        # we transform the shape from [batch_size, seq_size, hidden_dim]
        # to the actual [batch_seq_size, hidden_dim]

        lstm_output = tf.reshape(lstm_output, shape=(batch_size * sequence_length, self.hidden_dim))
        return lstm_output, hidden_state

    def forward_lstm(self, x, hidden):
        # note: No projection is used.
        # note: The outputs of the LSTM are the outputs of this module.
        lstm_out, hidden = x, [tf.squeeze(h) for h in hidden]
        for lstm in self.lstm_layers:
            lstm_out, *hidden = lstm(lstm_out, hidden)

        # DIFF: We apply layer norm to the gates.

        '''
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        '''

        return lstm_out, hidden

    def init_hidden_state(self, batch_size=1):
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        """
        hidden = (tf.zeros((self.n_layers, batch_size, self.hidden_dim)),
                  tf.zeros((self.n_layers, batch_size, self.hidden_dim)))
        return hidden

    def call(self, inputs, training=None, mask=None):
        embedded_scalar, embedded_entity, embedded_spatial, batch_size, sequence_length, hidden_state = inputs
        return self.forward(embedded_scalar, embedded_entity, embedded_spatial, batch_size=batch_size,
                            sequence_length=sequence_length, hidden_state=hidden_state)


def test():
    print("This is a test!") if debug else None


if __name__ == '__main__':
    test()
