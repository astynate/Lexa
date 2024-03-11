import os
import pickle

def load_dataset(path: str) -> list:
    with open(path, encoding='utf-8') as data:
        dataset = data.read().lower()

    return dataset

class LexaTokenizer:

    def __init__(self, path: str, **kwargs) -> None:

        if os.path.exists(path):

            with open(path, 'rb') as f:
                self.dictionary = pickle.load(f)
        
        else:

            text = self.prepare(kwargs.get('text'))
            self.dictionary = set()

            self.set_default(text)
            self.add_pairs(text)
            self.dictionary = list(self.dictionary)

            with open(path, 'wb') as f:
                pickle.dump(self.dictionary, f)

        print(f'LexaTokenizer created, number of tokens in dictionary: {len(self.dictionary)}')

    def prepare(self, text: str) -> str:
        return text.lower()

    def set_default(self, text) -> None:
        """
        Given a text, splits it into words and adds each symbol 
        from each word to the dictionary.
        """
        for word in text.split():
            self.dictionary.update(set(word))

        self.dictionary.add(' ')

    def add_pairs(self, text: str) -> None:

        pairs = set()

        to_replace = [',', ' ', '!', '?', '-', '\n', ')', '(' '[', ']', ':', "'"]
        to_replace += [str(i) for i in range(9)]

        for char in to_replace:
            text = text.replace(char, '')

        for i in range(len(text) - 1):
            pair = text[i:i + 2]
            pairs.add(pair)

        for pair in pairs:
            self.dictionary.add(pair)

    def get_full_sequence(self, text: str) -> list:

        text = text.lower().split()
        sequence = []

        for word in text:

            start = 0
            end = len(word) + 1

            while start < end:

                end -= 1
                
                if word[start:end] in self.dictionary:

                    sequence.append(self.dictionary.index(word[start:end]))
                    start = end
                    end = len(word) + 1
            
            sequence.append(self.dictionary.index(' '))

        return sequence
    
    def get_text(self, sequences: list) -> str:
        return ''.join(self.dictionary[i] for i in sequences)
    
    def get_dimension(self) -> int:
        return len(self.dictionary)
    
    def get_sequences(self, text: str) -> list:
        sequences = self.get_full_sequence(text)
        
        if len(sequences) > 50:
            return sequences[:50]
        elif len(sequences) < 50:
            return sequences + [0] * (50 - len(sequences))
        else:
            return sequences

if __name__ == '__main__':

    dataset = load_dataset('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/russian.txt')
    tokenizer = LexaTokenizer('D:/Exider Company/Lexa/Lexa AI/Assistant/models/lexa_tokenizer.pickle' , text=dataset)
    
    tokens = tokenizer.get_full_sequence('привет')
    original = tokenizer.get_text(tokens)

    print(tokens)
    print(original)