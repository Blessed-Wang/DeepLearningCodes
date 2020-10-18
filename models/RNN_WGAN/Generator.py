import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SequenceGenerator(keras.Model):
    def __init__(self, vocab_size, embedding_size, gru_layers, units, name='GRU_Generator'):
        super(SequenceGenerator, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.gru_layers = gru_layers
        self.units = units

        self.embed = layers.Dense(embedding_size, use_bias=False)
        self.cells = []
        for layer in range(gru_layers):
            self.cells.append(layers.GRUCell(units[layer], recurrent_dropout=0.2))
        self.dense = layers.Dense(vocab_size)
    
    def call(self, inputs, training=None, mask=None, initial_state=None):
        s_i = inputs
        hidden_state = initial_state
        s_i = self.embed(s_i)
        for gru_cell in self.cells:
            o, hidden_state = gru_cell(s_i, [hidden_state])
            s_i = o
            hidden_state = hidden_state[0]
        s_i = self.dense(s_i)
        s_i = tf.nn.softmax(s_i)
        return s_i, hidden_state
    
    def get_config(self):
        return super(SequenceGenerator, self).get_config()
