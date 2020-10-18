import tensorflow as tf
from tensorflow.keras import layers


class DownSamplingBlock(layers.Layer):
    def __init__(self, filters, kernel_size, apply_batch_normalization=True, name='down_sample'):
        super(DownSamplingBlock, self).__init__(name=name)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=2, padding='SAME',
            kernel_initializer=self.initializer,
            use_bias=False)
        self.batch_normalization = layers.BatchNormalization()
        self.apply_batch_normalization = apply_batch_normalization

    def build(self, input_shape):
        super(DownSamplingBlock, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv2d(x)
        if self.apply_batch_normalization:
            x = self.batch_normalization(x)
        x = tf.nn.leaky_relu(x)
        return x

    def get_config(self):
        config = super(DownSamplingBlock, self).get_config()
        config.update({
            'name': self.name,
            'is_apply_batchnorm': self.apply_batch_normalization
        })
        return config