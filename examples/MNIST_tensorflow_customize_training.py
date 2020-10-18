import numpy as np
import tensorflow as tf
from tensorflow import keras, losses, optimizers
from tensorflow.keras import layers

BATCH_SIZE = 300
EPOCHS = 30
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


def normalize(image, label):
    image = image.astype(np.float32)
    image = image / 255.
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).shuffle(1024).cache()
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(1024).batch(BATCH_SIZE)


model = keras.Sequential()
model.add(layers.Input([28, 28]))
model.add(layers.Reshape([28, 28, 1]))
model.add(layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='SAME'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='VALID'))
model.add(layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='SAME'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='VALID'))
model.add(layers.Flatten())
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

loss_object = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()
train_loss = tf.metrics.Mean(name='train_loss')
train_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
test_loss = tf.metrics.Mean(name='test_loss')
test_acc = tf.metrics.SparseCategoricalAccuracy(name='test_acc')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_acc(labels, predictions)


for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    for image, label in train_ds:
        train_step(image, label)

    for image, label in test_ds:
        test_step(image, label)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1, train_loss.result(), train_acc.result() * 100, test_loss.result(), test_acc.result() * 100))
