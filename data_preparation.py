import pandas as pd
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
wordnet_lemmatizer = WordNetLemmatizer()
import string
punctuations = string.punctuation
from sklearn import preprocessing
from gensim.models import Word2Vec
from sklearn.model_selection import KFold

# Text PreProcessing
# Remove extra whitespace
# Tokenize
# Remove punctuation, stopwords, convert to lower case
# Lemmatize
# word to vector


def key_feature(url):
    try:
        filename = './key_features/' + str(url) + '.txt'
        f = open(filename, "r")
        text = f.read()
        return text
    except Exception as e:
        raise e


def text_preprocessing(text):
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


def text_tokenizer():
    df['key_features'] = df['url'].map(key_feature)
    df['text'] = df['address'] + df['key_features']
    df['text'] = df['text'].map(text_preprocessing)
    return df


def change_unit(price):
    price = price / 1000
    return price


def take_length(text):
    return len(text)


if __name__ == '__main__':

    df = pd.read_csv("data_total_dl.csv")
    text_tokenizer()
    df['text_length'] = df['text'].map(take_length)
    input_text = [i for i in df['text']]
    skipgram_model = Word2Vec(input_text, min_count=1, size=25, workers=3, window=3, sg=1)
    skipgram_model.save("skipgram.model")
    cbow_model = Word2Vec(input_text, min_count=1, size=25, workers=3, window=3, sg=0)
    cbow_model.save("cbow.model")

    numeric_attrs = ['size', 'avg_price', 'bedrooms', 'bathrooms', 'receptions']
    for i in numeric_attrs:
        scaler = preprocessing.StandardScaler()
        df[i] = scaler.fit_transform(df[i].to_numpy().reshape(-1, 1))
    df['price'] = df['price'].map(change_unit)
    df.to_pickle("./data.pkl")
    df.to_csv("data.csv")

    kfold = KFold(n_splits=10, shuffle=True, random_state=20)
    cvscores = []
    fold = 1
    for train, test in kfold.split(df):
        trainDF = pd.DataFrame(df.loc[train])
        testDF = pd.DataFrame(df.loc[test])

        trainDF.to_pickle("./train_test_data/train_kfold" + str(fold) + ".pkl")
        testDF.to_pickle("./train_test_data/test_kfold" + str(fold) + ".pkl")
        fold += 1
