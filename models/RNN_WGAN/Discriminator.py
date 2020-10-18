from tensorflow import keras
from tensorflow.keras import layers


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = keras.Sequential([
            layers.Input([256, 256]),
            layers.Reshape([256, 256, 1]),
            layers.Conv2D(64, 3, 2),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        return super(Discriminator, self).get_config()