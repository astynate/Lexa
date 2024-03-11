from __model__ import *

BASE_PATH = 'D:/Exider Company/Lexa/Lexa AI/Assistant'
lexa = Lexa(50, BASE_PATH + '/models/lexa_tokenizer.pickle', 1, path=BASE_PATH + '/models/lexa.keras')

for i in range(5):

    context = lexa.tokenizer.get_text([np.random.randint(1, 30)])

    print(str(i))
    print(context, end=' ')

    for k in range(20):
        
        generated_word = lexa(context)
        context += generated_word

    print(context)