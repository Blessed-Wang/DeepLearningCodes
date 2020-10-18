import tensorflow as tf
from tensorflow import keras, losses, optimizers
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
# load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# normalize
X_train = X_train.astype(np.float32)
X_train = X_train / 255.
X_test = X_test.astype(np.float32)
X_test = X_test / 255.

BATCH_SIZE = 512
EPOCHS = 20

# build model
model = keras.Sequential()
model.add(layers.Input([28, 28]))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

loss = losses.sparse_categorical_crossentropy
optimizer = optimizers.Adam()
metrics = ['accuracy']
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
train_history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, callbacks=callbacks)
model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(train_history.history['loss'], label='loss')
plt.plot(train_history.history['val_loss'], label='val_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(122)
plt.plot(train_history.history['accuracy'], label='accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
