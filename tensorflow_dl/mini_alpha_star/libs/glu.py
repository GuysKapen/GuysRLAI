import tensorflow as tf
import tensorflow.keras.layers as layers

debug = True


class GLU(tf.keras.Model):
    def __init__(self, input_size=384, context_size=1024,
                 output_size=1024):
        super(GLU, self).__init__()
        self.fc1 = layers.Dense(input_shape=(1, context_size), units=input_size)
        self.fc2 = layers.Dense(output_size)
        self.sigmoid = tf.keras.layers.Activation(activation='sigmoid')

    def forward(self, x, context):
        # context shape: b x context_size
        gate = self.sigmoid(self.fc1(context))
        # gate shape: b x input_size

        gated_input = gate * x
        # gated_input: b x input_size

        output = self.fc2(gated_input)
        # output: b x output_size
        return output

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        x, context = inputs
        return self.forward(x, context)


def test():
    context = tf.random.normal(shape=(5, 32))
    x = tf.random.normal(shape=(5, 16))
    output_size = 24

    model = GLU(input_size=x.shape[-1], context_size=context.shape[-1],
                output_size=output_size)

    print(f'context: {context.shape}') if debug else None
    print(f'x: {x.shape}') if debug else None

    output = model((x, context))
    print(f'output: {output.shape}') if debug else None


if __name__ == '__main__':
    test()
