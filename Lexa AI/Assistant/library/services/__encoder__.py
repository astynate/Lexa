import re

import re

def tokenize(text: str) -> list:
    tokens = re.findall(r'\b\w+\b', text)
    result = []

    for token in tokens:
        for char in token:
            result.append(ord(char.lower()))

            if len(result) >= 100:
                return result[:100]
            
        result.append(ord(' '))

    while len(result) < 100:
        result.append(0)

    return result


def detokenize(tokens: list) -> str:
    result = ''
    
    for token in tokens:
        result += chr(int(token))

    return result

if __name__ == '__main__':

    tokens = tokenize('Привет как дела hello beach')
    original_text = detokenize(tokens)

    print(tokens)
    print(original_text)