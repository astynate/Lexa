# BOOK_PATH = 'D:/[1] Exider Projects/Lexa/Lexa AI/Assistant/Толстой Лев. Война и мир. Книга 1 - royallib.ru.txt'
BOOK_PATH = 'D:/Exider Company/Lexa/Lexa AI/Assistant/book_one.txt'

with open(BOOK_PATH, 'r', encoding='utf-8') as book:

    book = book.read()

print("Book is loaded, words: " + str(len(book.lower().split())))