import tensorflow as tf

BATCH_SIZE = 256
BUFFER_SIZE = 50000
EPOCHS = 50
NOISE_DIM = 100
NUMS_EXAMPLES_TO_GENERATE = 16
SEED = tf.random.normal([NUMS_EXAMPLES_TO_GENERATE, NOISE_DIM])