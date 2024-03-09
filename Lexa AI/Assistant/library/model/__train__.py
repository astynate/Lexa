import os
import re
import tensorflow as tf
from library.data.vocabulary import *
from library.model.__model__ import *
from library.data.__dataset__ import *
from library.services.__encoder__ import tokenize, detokenize
import numpy as np

PATH = 'D:/Exider Company/Lexa/Lexa AI/Assistant/'
MODEL_PATH = PATH + 'model.h5'
SEQUENCE_LENGTH = 1
VOCAB_SIZE = 386659
END_OF_SENTANCE = ['.', '!', '?']

def prepare_data() -> np.array:

    dataset = book.lower()

    for end_point in END_OF_SENTANCE:
        dataset = dataset.replace(end_point, '<end>')

    dataset = re.sub("[^а-яА-Я <end>]", "", dataset)
    dataset = dataset.split('<end>')

    input_values = []
    valid_values = []

    for sentance in dataset:

        if (len(sentance) > 1 and sentance.split()[-1] in ru):

            print(' '.join(sentance.split()[:-1]))

            input_values.append([tokenize(' '.join(sentance.split()[:-1]))])

            value = [0 for i in range(VOCAB_SIZE)]
            index = ru.index(sentance.split()[-1])
            value[index] = 1
            
            valid_values.append(value)

    np.save(PATH + 'input_values.npy', np.array(input_values).reshape(6, 100))
    np.save(PATH + 'valid_values.npy', np.array(valid_values).reshape((6, VOCAB_SIZE)))

    return [input_values, valid_values]

def train() -> None:

    model = None

    if os.path.exists(MODEL_PATH) and os.path.isfile(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)

    else:
        model = Lexa(VOCAB_SIZE, 10, 1024, 256, 4, 0.8, SEQUENCE_LENGTH).model

    input_values = np.load(PATH + 'input_values.npy', allow_pickle=True)
    valid_values = np.load(PATH + 'valid_values.npy', allow_pickle=True)

    # valid_values = np.squeeze(valid_values, axis=1)

    # print(input_values)
    # print(valid_values)

    # print(input_values[0])
    # print(valid_values[0])

    # for i in input_values:

    #     print(i[0])
    #     print(detokenize(i[0]))

    # model.fit(input_values, valid_values, epochs=10)

    # model.save(MODEL_PATH)

    output = model.predict(np.array(tokenize('здравствуй мать в канаве')).reshape(1, 100)).tolist()
    
    print(np.array(tokenize('здравствуй мать в канаве')).reshape(1, 100))
    print(np.array(output).shape)

    for i in range(SEQUENCE_LENGTH):

        max_element = max(output[0][i])
        index = output[0][i].index(max_element)

        print(detokenize(tokenize('здравствуй мать в канаве')), end=' - ')
        print(ru[index], end=' ')

if __name__ == '__main__':

    # prepare_data()
    train()