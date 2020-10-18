from models.xu_net.XuNet import XuNet
from models.xu_net.block import SampleBlock
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

from models.xu_net.hyperparameters import BATCH_SIZE

group1 = SampleBlock(filters=8, kernel_size=5, apply_abs=True, activation=keras.activations.tanh, name='group1')
y = group1(tf.random.normal([1, 512, 512, 1]))
print(y.shape)

group2 = SampleBlock(filters=16, kernel_size=5, activation=keras.activations.tanh, name='group2')
z = group2(y)
print(z.shape)

model = XuNet()
output = model(tf.random.normal([BATCH_SIZE, 512, 512, 1]))
print(output.shape)
model.xu_net.summary()