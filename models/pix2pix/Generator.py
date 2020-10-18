import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from blocks.DownsampleBlock import DownSamplingBlock
from blocks.UpSamplingBlock import UpSamplingBlock
from models.pix2pix.hyperparameters import OUTPUT_CHANNELS


class Generator(keras.Model):
    def __init__(self, output_channels = OUTPUT_CHANNELS, name='unet'):
        super(Generator, self).__init__(name=name)
        self.output_channels = output_channels
        self.down_sample_stack = [
            # Input: [bs, 256, 256, 3]
            DownSamplingBlock(64, 4, apply_batch_normalization=False),  # [bs, 128, 128, 64]
            DownSamplingBlock(128, 4),  # [bs, 64, 64, 128]
            DownSamplingBlock(256, 4),  # [bs, 32, 32, 256]
            DownSamplingBlock(512, 4),  # [bs, 16, 16, 512]
            DownSamplingBlock(512, 4),  # [bs, 8,  8,  512]
            DownSamplingBlock(512, 4),  # [bs, 4,  4,  512]
            DownSamplingBlock(512, 4),  # [bs, 2,  2,  512]
            DownSamplingBlock(512, 4)   # [bs, 1,  1,  512]
        ]
        self.up_sample_stack = [
            UpSamplingBlock(512, 4, apply_dropout=True, drop_rate=0.5),  # [bs, 2,  2,  1024]
            UpSamplingBlock(512, 4, apply_dropout=True, drop_rate=0.5),  # [bs, 4,  4,  1024]
            UpSamplingBlock(512, 4, apply_dropout=True, drop_rate=0.5),  # [bs, 8,  8,  1024]
            UpSamplingBlock(512, 4),  # [bs, 16, 16, 1024]
            UpSamplingBlock(256, 4),  # [bs, 32, 32, 512]
            UpSamplingBlock(128, 4),  # [bs, 64, 64, 256]
            UpSamplingBlock(64, 4)    # [bs, 128,128,128]
        ]
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.last = layers.Conv2DTranspose(output_channels, 4, strides=2, padding='SAME', kernel_initializer=self.initializer, activation='tanh')
        self.concat = layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        x = inputs
        skips = []
        for down in self.down_sample_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_sample_stack, skips):
            x = up(x)
            x = self.concat([x, skip])
        x = self.last(x)
        return x

    def build(self, input_shape):
        super(Generator, self).build(input_shape)

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({
            'out_channels': self.output_channels
        })
        return config