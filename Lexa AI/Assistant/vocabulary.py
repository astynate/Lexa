# import requests

# base_url = 'https://github.com/LussRus/Rus_words/tree/master/UTF8/txt'
# paths = ['adject', 'nouns', 'raw', 'verbs']

# file_path_ru = 'D:/[1] Exider Projects/Lexa/Lexa AI/Assistant/russian.txt'

# for path in paths:

#     response = requests.get(base_url + path + 'summary.txt')
#     text = response.content.decode('cp1251')

#     print(text)

#     with open(file_path_ru, 'a') as ru:
#         ru.write(text.encode('utf-8'))

PATH_RU = 'D:/[1] Exider Projects/Lexa/Lexa AI/Assistant/russian.txt'

with open(PATH_RU, 'r', encoding='utf-8') as file:
    ru = file.read().split()

print("The Russian language dictionary is loaded: " + str(len(ru)))