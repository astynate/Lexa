import os
import tensorflow as tf
import numpy as np
from __model__ import Lexa

BASE_PATH = 'D:/Exider Company/Lexa/Lexa AI/Assistant'
EMBEDDING_DIMENTION = 40
NUMBER_OF_HEADES = 2
FEED_FORWARD_DIMENTION = 32

def train() -> None:

    lexa = Lexa(50, BASE_PATH + '/models/lexa_tokenizer.pickle', 1, 25)
    model = lexa.model

    # text = open('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/book_one.txt', 'rb').read().decode(encoding='utf-8')

    # tokenizer = lexa.tokenizer(char_level=True)
    # tokenizer.fit_on_texts(text)

    # sequences = tokenizer.texts_to_sequences([text])[0]
    # sequences = np.array([sequences[i:i+51] for i in range(len(sequences)-50)])

    # train_data, train_labels = sequences[:, :-1], sequences[:, -1]
    # train_data = tf.one_hot(train_data, depth=len(tokenizer.word_index) + 1)
    # train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(tokenizer.word_index) + 1)

    # model.fit(train_data, train_labels, batch_size=32, epochs=1)

    print(lexa('привет как дела'))

if __name__ == '__main__':

    train()