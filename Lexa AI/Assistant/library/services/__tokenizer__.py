import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def load_dataset(path: str) -> list:

    with open(path, 'r', encoding='utf-8') as data:
        dataset = data.read().lower()

    return dataset.split()

class LexaTokenizer:

    def __init__(self, path: str, **kwargs) -> None:

        self.path = path + 'lexa_tokenizer.pickle'

        if os.path.exists(self.path):

            with open(self.path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:

            dataset = kwargs.get('dataset')

            if dataset is not None:

                self.tokenizer = Tokenizer()
                self.tokenizer.fit_on_texts(dataset)

                with open(self.path, 'wb') as handle:

                    pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            else:
                raise ValueError("No existing tokenizer found and no dataset provided to train a new one.")

    def get_sequences(self, text: str) -> list:
        return self.tokenizer.texts_to_sequences([text])[0]

    def get_text(self, sequences: list) -> str:
        return self.tokenizer.sequences_to_texts([sequences])[0]

if __name__ == '__main__':

    model_path: str = 'D:/Exider Company/Lexa/Lexa AI/Assistant/models/'
    dataset = load_dataset('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/russian.txt')

    tokenizer = LexaTokenizer(model_path, dataset)

    test_token = tokenizer.get_sequences('привет как дела')
    original_text = tokenizer.get_text(test_token)

    print(test_token)
    print(original_text)