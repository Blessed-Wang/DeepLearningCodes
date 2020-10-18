from tensorflow.keras import layers
import tensorflow as tf


class SampleBlock(layers.Layer):
    def __init__(self, filters, kernel_size=5, apply_abs=False,
                 pool_layer=layers.AveragePooling2D(5, 2, 'SAME'),
                 activation=None,
                 name='sample_block'):
        super(SampleBlock, self).__init__(name=name)
        self.apply_abs = apply_abs
        self.initializer = tf.random_normal_initializer(0., 0.01)
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='SAME', kernel_initializer=self.initializer, use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.activation = activation
        self.average_pool = pool_layer

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.conv(x)
        if self.apply_abs:
            x = tf.abs(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.average_pool(x)
        return x

    def get_config(self):
        config = super(SampleBlock, self).get_config()
        config.update()
        return config
