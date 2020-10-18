import tensorflow as tf
from tensorflow.keras import layers

from blocks.DownsampleBlock import DownSamplingBlock


class Discriminator(tf.keras.Model):
    def __init__(self, name='discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.concat = layers.Concatenate()
        self.down1 = DownSamplingBlock(64, 4, apply_batch_normalization=False)
        self.down2 = DownSamplingBlock(128, 4)
        self.down3 = DownSamplingBlock(256, 4)
        self.zero_pad = layers.ZeroPadding2D()
        self.zero_pad2 = layers.ZeroPadding2D()
        self.conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=self.initializer, use_bias=False)
        self.batch_norm = layers.BatchNormalization()
        self.last = layers.Conv2D(1, 4, strides=1, kernel_initializer=self.initializer)

    def call(self, inputs):
        inp, tar = inputs
        x = self.concat([inp, tar])
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.zero_pad(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = tf.nn.leaky_relu(x)
        x = self.zero_pad2(x)
        x = self.last(x)
        return x
    
    def get_config(self):
        return super(Discriminator, self).get_config()