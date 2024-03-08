from __model__ import *
from vocabulary import ru
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequence_length = 1
vocab_size = 386659

phrases = ['Привет']
model = Lexa(vocab_size, 10, 1024, 256, 4, 0.8, sequence_length).model

for phrase in phrases:

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts([phrase])
    input_sequence = tokenizer.texts_to_sequences([phrase])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=sequence_length, padding='post')

    print(np.array(padded_input_sequence).shape)

    output = model.predict(np.array(padded_input_sequence)).tolist()

    print(np.array(output).shape)

    for i in range(sequence_length):

        max_element = max(output[0][i])
        index = output[0][i].index(max_element)

        print(ru[index], end=' ')