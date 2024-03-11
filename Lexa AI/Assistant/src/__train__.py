import tensorflow as tf
import numpy as np
from __model__ import Lexa

BASE_PATH = 'D:/Exider Company/Lexa/Lexa AI/Assistant'
EMBEDDING_DIMENTION = 40
NUMBER_OF_HEADES = 2
FEED_FORWARD_DIMENTION = 32

def train() -> None:

    lexa = Lexa(50, BASE_PATH + '/models/lexa_tokenizer.pickle', 1, path=BASE_PATH + '/models/lexa.keras')
    text = open('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/book_one.txt', 'rb').read().decode(encoding='utf-8')

    sequences = lexa.tokenizer.get_full_sequence(text)
    sequences = np.array([sequences[i:i+51] for i in range(len(sequences)-50)])

    print(sequences)

    train_data, train_labels = sequences[:, :-1], sequences[:, -1]
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=lexa.tokenizer.get_dimension())

    print(train_data.shape)
    print(train_labels.shape)

    for _ in range(1):

        lexa.model.fit(train_data, train_labels, batch_size=64, epochs=1)
        lexa.model.save(BASE_PATH + '/models/lexa.keras')

        context = lexa.tokenizer.get_text([np.random.randint(1, 30)])
        print(context)

        for i in range(1):
            
            print(str(i) + ' |' + context + '|')
            generated_word = lexa(context)

            context += generated_word

if __name__ == '__main__':

    train()