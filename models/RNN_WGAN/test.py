import tensorflow as tf
from tensorflow import keras, losses, optimizers, metrics
from tensorflow.keras import layers

from models.RNN_WGAN.Generator import SequenceGenerator

model = SequenceGenerator(256, 100, 1, [1024, ])
output = model(tf.random.normal([10, 1]), initial_state=tf.zeros([10, 1024]))
print(output[0].shape, output[1].shape)