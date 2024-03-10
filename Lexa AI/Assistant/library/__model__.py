import tensorflow as tf
import numpy as np
import os
from __tokenizer__ import LexaTokenizer
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Lexa:

    def __init__(self, input_shape, tokenizer_path, heads, ffd_dim, **kwargs) -> None:

        self.tokenizer = LexaTokenizer(tokenizer_path)
        self.embed_dim = len(self.tokenizer.tokenizer.word_index) + 1

        if kwargs.get('path') is not None and os.path.exists(kwargs.get('path')):

            self.model = tf.keras.models.load_model(kwargs.get('path'))

        else:

            inputs = tf.keras.Input(shape=(input_shape, self.embed_dim))
            x = TransformerBlock(self.embed_dim, heads, ffd_dim)(inputs, training=True)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(self.embed_dim, activation="softmax")(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

        print("A Lexa model was created, the number of parameters: " + str(self.model.count_params()))

    def __call__(self, text: str) -> str:

        tokens = self.tokenizer.get_sequences(text)
        
        predicted_label = self.model.predict(tokens)
        predicted_word = self.tokenizer.get_text([[np.argmax(predicted_label[0])]])

        return predicted_word

# model.fit(train_data, train_labels, batch_size=32, epochs=5)

# Load the dataset
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Preprocess the data

# Compile the model

# Fit the model to the training data

# input_string = "To be or not to be, that is the question:"

# Pad the input string with spaces if it's shorter than 50 characters
# input_string = input_string.ljust(50)

# # Convert the input string to sequences and one-hot encode it
# input_data = tf.one_hot(tokenizer.texts_to_sequences([input_string])[0], depth=len(tokenizer.word_index)+1)

# # Use the model to make predictions