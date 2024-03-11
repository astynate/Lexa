import tensorflow as tf
import numpy as np
import os
from tokenizer import LexaTokenizer
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

class PositionEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return tf.cast(position, tf.float32) * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])

        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + tf.cast(self.pos_encoding[:, :tf.shape(inputs)[1], :], dtype=tf.float16)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), 
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class Lexa:

    def __init__(self, input_shape, tokenizer_path, heads, **kwargs) -> None:

        self.tokenizer = LexaTokenizer(tokenizer_path)
        self.embed_dim = self.tokenizer.get_dimension()

        print(self.embed_dim)

        if kwargs.get('path') is not None and os.path.exists(kwargs.get('path')):

            self.model = tf.keras.models.load_model(kwargs.get('path'))

        else:

            inputs = tf.keras.Input(shape=(input_shape,))
            embedding_layer = layers.Embedding(input_dim=self.tokenizer.get_dimension(), output_dim=self.embed_dim)
            x = embedding_layer(inputs)
            x = PositionEncoding(self.embed_dim, self.embed_dim)(x)
            x = TransformerBlock(self.embed_dim, heads, self.embed_dim)(x, training=True)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(self.embed_dim, activation="softmax")(x)

            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

        print("A Lexa model was created, the number of parameters: " + str(self.model.count_params()))

    def __call__(self, text: str) -> str:

        tokens = self.tokenizer.get_sequences(text)
        tokens = np.array(tokens).reshape(1, 50)
        
        predicted_label = self.model.predict(np.array(tokens))
        predicted_word = self.tokenizer.get_text([[np.argmax(predicted_label[0])]][0])

        return predicted_word

if __name__ == '__main__':

    lexa = Lexa(50, 'D:/Exider Company/Lexa/Lexa AI/Assistant' + '/models/lexa_tokenizer.pickle', 1)