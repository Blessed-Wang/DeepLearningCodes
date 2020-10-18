import tensorflow as tf
from tensorflow.keras import layers


class UpSamplingBlock(layers.Layer):
    def __init__(self, filters, kernel_size, apply_dropout=False, name='up_sample', drop_rate=0.5):
        super(UpSamplingBlock, self).__init__(name=name)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.apply_dropout = apply_dropout
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv2d_transpose = layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=2, padding='SAME',
            kernel_initializer=self.initializer,
            use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(drop_rate)

    def build(self, input_shape):
        super(UpSamplingBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv2d_transpose(x)
        x = self.batch_norm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        x = tf.nn.relu(x)
        return x

    def get_config(self):
        config = super(UpSamplingBlock, self).get_config()
        config.update({
            'name': self.name,
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config
