import os
import pickle

def load_dataset(path: str) -> list:
    with open(path, encoding='utf-8') as data:
        dataset = data.read().lower()

    return dataset

class LexaTokenizer:

    def __init__(self, path: str, special_symbols: list, **kwargs) -> None:

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.token_list = pickle.load(f)
        else:
            self.token_list = self.split_text(kwargs.get('dataset').replace(',', '').replace('.', ''), special_symbols)

            with open(path, 'wb') as f:
                pickle.dump(self.token_list, f)

        print('A Lexs Tokenizer object was created with its dimension: ' + str(self.get_dimension()))

    def split_text(self, text, args):
        
        words = text.lower().split()
        unique_subwords = set()

        for word in words:
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    unique_subwords.add(word[i:j])

        return list(unique_subwords) + args

    def get_full_sequence(self, text: str) -> list:

        text = text.lower().split()
        sequence = []

        for word in text:

            for sub_word in word:

                for i in sub_word:
                    
                    if i in self.token_list:

                        sequence.append(self.token_list.index(i))
                        print(f'|{i}|')
            
            sequence.append(self.token_list.index(' '))

        return sequence

    def get_sequences(self, text: str) -> list:
        sequences = self.tokenizer.encode(text)
        
        if len(sequences) > 50:
            return sequences[:50]
        elif len(sequences) < 50:
            return sequences + [0] * (50 - len(sequences))
        else:
            return sequences

    def get_text(self, sequences: list) -> str:
        return ''.join(self.token_list[i] for i in sequences)
    
    def get_dimension(self) -> int:
        return len(self.token_list)

if __name__ == '__main__':

    model_path: str = 'D:/Exider Company/Lexa/Lexa AI/Assistant/models/lexa.txt'
    dataset = load_dataset('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/russian.txt')

    tokenizer = LexaTokenizer(model_path, [' ', '.', '!', '?', '-', ','], dataset=dataset)

    test_token = tokenizer.get_full_sequence('когда Элария, верховная')
    original_text = tokenizer.get_text(test_token)

    print(test_token)
    print(original_text)

    print("Количество токенов в токенизаторе:", tokenizer.get_dimension())