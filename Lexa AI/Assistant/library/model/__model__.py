import tensorflow as tf
import numpy as np
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

embed_dim = 40
num_heads = 2
ff_dim = 32

inputs = layers.Input(shape=(50, embed_dim))
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(inputs, training=True)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(40, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(40, activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.fit(train_data, train_labels, batch_size=32, epochs=5)

# Load the dataset
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

text = open('D:/Exider Company/Lexa/Lexa AI/Assistant/book_one.txt', 'rb').read().decode(encoding='utf-8')

# Preprocess the data
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(text)

sequences = tokenizer.texts_to_sequences([text])[0]
sequences = np.array([sequences[i:i+51] for i in range(len(sequences)-50)])

train_data, train_labels = sequences[:, :-1], sequences[:, -1]
train_data = tf.one_hot(train_data, depth=len(tokenizer.word_index)+1)
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(tokenizer.word_index)+1)

# Compile the model
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

# Fit the model to the training data
# model.fit(train_data, train_labels, batch_size=32, epochs=1)

input_string = "To be or not to be, that is the question:"

# Pad the input string with spaces if it's shorter than 50 characters
input_string = input_string.ljust(50)

# # Convert the input string to sequences and one-hot encode it
input_data = tf.one_hot(tokenizer.texts_to_sequences([input_string])[0], depth=len(tokenizer.word_index)+1)

print(tokenizer.texts_to_sequences([input_string]))
print(tokenizer.texts_to_sequences([input_string])[0])
print(input_data)
print(len(tokenizer.word_index))

# # Reshape the input data to match the expected input shape of the model
# input_data = np.reshape(input_data, (1, 50, 40))

# # Use the model to make predictions
# predicted_label = model.predict(input_data)
# predicted_word = tokenizer.sequences_to_texts([[np.argmax(predicted_label[0])]])[0]

# print(f'Input: {input_string}')
# print(f'Predicted next word: {predicted_word}')