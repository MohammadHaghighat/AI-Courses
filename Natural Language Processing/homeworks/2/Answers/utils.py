from collections import Counter
import re
def preprocess(text):
    text = re.sub(r'\.', ' <PERIOD> ', text)
    text = re.sub(r',', ' <COMMA> ', text)
    text = re.sub(r'\?', ' <QUESTION> ', text)
    text = re.sub(r'!', ' <EXCLAMATION> ', text)
    text = re.sub(r';', ' <SEMICOLON> ', text)
    text = re.sub(r':', ' <COLON> ', text)
    text = re.sub(r'"', ' <QUOTE> ', text)
    text = re.sub(r'\(', ' <LEFTPAREN> ', text)
    text = re.sub(r'\)', ' <RIGHTPAREN> ', text)
    text = re.sub(r'\[', ' <LEFTBRACKET> ', text)
    text = re.sub(r'\]', ' <RIGHTBRACKET> ', text)
    text = re.sub(r'\-', ' <HYPHEN> ', text)
    text = re.sub(r'/', ' <SLASH> ', text)
    words = text.lower().split()
    word_counts = Counter(words)

    final_words = []
    for word in words:
        if word_counts[word] > 5:
            final_words.append(word)
    return final_words
from collections import Counter

def create_lookup_tables(words):
    word_counts = Counter(words)

    sorted_words = []
    for word, frequency in word_counts.most_common():
        sorted_words.append(word)
        
    word_to_int = {}
    for i, word in enumerate(sorted_words):
        word_to_int[word] = i
        
    int_to_word = {}
    for word, i in word_to_int.items():
        int_to_word[i] = word

    return word_to_int, int_to_word
