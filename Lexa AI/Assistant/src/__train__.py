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

    full_sequence = lexa.tokenizer.get_full_sequence(text)
    sequences = [full_sequence[i:i+np.random.randint(2, 51)] for i in range(len(full_sequence)-50)]

    input_values = []
    correct_answers = []

    for sub_sequence in sequences:

        correct = lexa.tokenizer.get_text([sub_sequence[-1]])

        if correct != '\ufffd':

            value = lexa.tokenizer.get_sequences(lexa.tokenizer.get_text(sub_sequence[:-1]))
            token_probability = tf.keras.utils.to_categorical(sub_sequence[-1], num_classes=lexa.tokenizer.get_dimension())

            input_values.append(value)
            correct_answers.append(token_probability)

    print(np.array(input_values).shape)
    print(np.array(correct_answers).shape)
    print(np.array(input_values))

    for i in input_values:

        print(i)

    # print(np.array(correct_answers))

    # for _ in range(1):

    #     lexa.model.fit(np.array(input_values), np.array(correct), batch_size=64, epochs=1)
    #     lexa.model.save(BASE_PATH + '/models/lexa.keras')

    #     context = lexa.tokenizer.get_text([np.random.randint(1, 30)])

    #     print(context)

    #     for i in range(1):
            
    #         print(str(i) + ' |' + context + '|')
    #         generated_word = lexa(context)

    #         context += generated_word

if __name__ == '__main__':

    train()