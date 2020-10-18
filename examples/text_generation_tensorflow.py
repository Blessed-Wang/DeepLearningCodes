import tensorflow as tf
from tensorflow import keras
from tensorflow import losses, metrics, optimizers
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                          split=('train', 'test'),
                                          with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
print(encoder.subwords[:20])

train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)

train_batch, train_labels = next(iter(train_data))
print(train_batch, train_labels)
embedding_dim = 32
model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_dim),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.summary()
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer=optimizers.Adam(), metrics=['accuracy'])
train_history = model.fit(train_batches, epochs=10, validation_data=test_batches, validation_steps=20)

