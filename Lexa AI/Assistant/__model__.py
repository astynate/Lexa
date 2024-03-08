import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def tokenize(phrase, vocab_size, sequence_length) -> np.array:

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([str(phrase)])
    input_sequence = tokenizer.texts_to_sequences([str(phrase)])
    
    return np.array(pad_sequences(input_sequence, maxlen=sequence_length, padding='post'))

def convert_token_to_letters(token, vocab_size):

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    word_index = tokenizer.word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    return reverse_word_index.get(token, "<OOV>")

def TransformerBlock(d_model, num_heads, units, dropout):

    inputs = keras.Input(shape=(None, d_model))
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    outputs = layers.Dense(units, activation='relu')(attention_output)
    outputs = layers.Dense(d_model)(outputs)
    outputs = layers.Dropout(dropout)(outputs)
    outputs = layers.LayerNormalization(epsilon=1e-6)(attention_output + outputs)

    return keras.Model(inputs=inputs, outputs=outputs)

def PositionalEncoding(max_position, d_model):

    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class Lexa:

    def __init__(self, vocab_size, num_layers, units, d_model, num_heads, dropout, sequence_length):

        self.inputs = tf.keras.Input(shape=(sequence_length,))
        self.embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(self.inputs)

        self.positional_encoding = PositionalEncoding(sequence_length, d_model)

        x = self.embeddings + self.positional_encoding

        for _ in range(num_layers):
            x = TransformerBlock(d_model, num_heads, units, dropout)(x)

        self.outputs = tf.keras.layers.Dense(vocab_size)(x)
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print("A Lexa model was created, the number of parameters: " + str(self.model.count_params()))

if __name__ == '__main__':

    model = Lexa(386660, 1, 5, 1, 4, 0.3, 100).model
    output = model(np.array([i for i in range(100)])).numpy().flatten().tolist()

    print(output.index(max(output)))