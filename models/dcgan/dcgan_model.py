import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Generator(keras.Model):
    def __init__(self, name='generator'):
        super(Generator, self).__init__(name=name)
        self.model = keras.Sequential([
            layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100, )),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape([7, 7, 256]),
            layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        return self.model(x)

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({'name': self.name})
        return config


class Discriminator(keras.Model):
    def __init__(self, name='discriminator'):
        super(Discriminator, self).__init__(name=name)
        self.model = keras.Sequential([
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        return self.model(x)
    
    def get_config(self):
        return super(Discriminator, self).get_config()