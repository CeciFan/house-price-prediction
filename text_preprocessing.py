import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
wordnet_lemmatizer = WordNetLemmatizer()
import string
punctuations = string.punctuation

from gensim.models import Word2Vec

# Text PreProcessing
# Remove extra whitespace
# Tokenize
# Remove punctuation, stopwords, convert to lower case
# Lemmatize
# word to vector


def nltk_tokenizer(url):
    try:
        filename = './key_features/' + str(url) + '.txt'
        f = open(filename, "r")
        text = f.read()
        text = text.replace("-", "").replace("/", " ")
        tokens = [word for word in word_tokenize(text) if word.isalpha()]
        tokens = list(filter(lambda t: t not in punctuations, tokens))
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        filtered_tokens = list(map(lambda token: wordnet_lemmatizer.lemmatize(token.lower()), filtered_tokens))
        filtered_tokens = list(filter(lambda t: t not in punctuations, filtered_tokens))
        return filtered_tokens
    except Exception as e:
        raise e


def dask_tokenizer(df):
    df['text'] = df['url'].map(nltk_tokenizer)
    return df


def combine_word_vector(words):
    vector = np.array([0] * 25)
    for word in words:
        vector = np.add(wordvectors.word_vec(word), vector)
    return vector


if __name__ == '__main__':
    df = pd.read_csv("data_total_dl.csv")
    df = dask_tokenizer(df)
    df.drop(['Unnamed: 0'], axis=1)
    input_text = [i for i in df['text']]
    model = Word2Vec(input_text, min_count=1, size=25, workers=3, window=3, sg=0)
    wordvectors = model.wv  # KeyedVectors Instance gets stored
    df['word_vectors'] = df['text'].map(combine_word_vector)
    df.to_csv("data_total_dl.csv")