import tensorflow as tf

from blocks.UpSamplingBlock import UpSamplingBlock

up_sample = UpSamplingBlock(
    filters=32,
    kernel_size=3,
    apply_dropout=True,
    drop_rate=0.5
)

up_sample.build([1, 32, 32, 3])
x = tf.random.normal([1, 32, 32, 3])
y = up_sample(x)
print(y.shape)