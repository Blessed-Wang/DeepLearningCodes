from tensorflow import keras
from tensorflow.keras import layers

from models.xu_net.block import SampleBlock


class XuNet(keras.Model):
    def __init__(self, name='xu_net'):
        super(XuNet, self).__init__(name=name)
        self.xu_net = keras.Sequential([
            layers.Input([512, 512]),
            layers.Reshape([512, 512, 1]),
            SampleBlock(filters=8, apply_abs=True, activation=keras.activations.tanh, name='group1'),
            SampleBlock(filters=16, activation=keras.activations.tanh, name='group2'),
            SampleBlock(filters=32, activation=keras.activations.relu, name='group3'),
            SampleBlock(filters=64, activation=keras.activations.relu, name='group4'),
            SampleBlock(filters=128, activation=keras.activations.relu, name='group5',
                        pool_layer=layers.GlobalAveragePooling2D()),
            layers.Flatten(),
            layers.Dense(2, activation='softmax')
        ])

    def call(self, inputs, training=None, mask=None):
        x = inputs
        return self.xu_net(x)
    
    def get_config(self):
        return super(XuNet, self).get_config()