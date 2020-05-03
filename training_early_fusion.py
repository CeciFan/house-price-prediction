import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Concatenate, Embedding
from keras.layers import Flatten, Reshape, Conv1D, MaxPooling1D, SimpleRNN
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def build_model(architecture, embeddings_dim):

    # add embedding layer
    input_emb = Input(shape=(20, ))
    input_num = Input(shape=(5, ))

    # words have already been trained
    e = Embedding(input_dim=len(embeddings_matrix),
                  output_dim=embeddings_dim,
                  weights=[embeddings_matrix],
                  input_length=20,
                  trainable=False)(input_emb)

    flatten = Flatten()(e)
    conc = Concatenate()([input_num, flatten])

    if architecture == 'mlp':
        main = Dense(100, activation='relu')(conc)
        main = Dropout(0.02)(main)
        main = Dense(100, activation='relu')(main)
        main = Dropout(0.02)(main)
        main = Dense(100, activation='relu')(main)
        main = Dropout(0.02)(main)
        main = Dense(100, activation='relu')(main)
        main = Dropout(0.02)(main)

    elif architecture == 'cnn':
        main = Reshape((505, 1))(conc)
        main = Conv1D(100, 3, strides=1, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Dropout(0.02)(main)
        main = Conv1D(100, 3, strides=1, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Dropout(0.02)(main)
        main = Conv1D(100, 3, strides=1, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Dropout(0.02)(main)
        main = Flatten()(main)

    elif architecture == 'rnn':
        main = Reshape((505, 1))(conc)
        # simpleRNN
        main = Conv1D(100, 3, strides=1, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Dropout(0.02)(main)
        main = SimpleRNN(100)(main)
        main = Dropout(0.02)(main)

    else:
        print('Error: Model type not found.')

    main_output = Dense(1)(main)
    model = Model(inputs=[input_emb, input_num], outputs=[main_output], name=architecture)
    sgd = optimizers.SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['mae'])

    return model


# give each word an index
def toindex(words):
    data = []
    for word in words:
        try:
            data.append(word_index[word])
        except:
            continue

    return data


if __name__ == '__main__':

    # import trained words vector
    model = Word2Vec.load("skipgram.model")

    wordvectors = model.wv
    vocab_list = [word for word, Vocab in wordvectors.vocab.items()]
    word_index = {" ": 0}
    word_vector = {}
    embedding_dim = model.vector_size
    embeddings_matrix = np.zeros((len(vocab_list) + 1, embedding_dim))
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i + 1
        word_vector[word] = wordvectors[word]
        embeddings_matrix[i + 1] = wordvectors[word]

    cvscores = []

    for fold in range(1, 11):
        train_data = pd.read_pickle("./train_test_data/train_kfold" + str(fold) + ".pkl")
        test_data = pd.read_pickle("./train_test_data/test_kfold" + str(fold) + ".pkl")

        words_train = pad_sequences(train_data['text'].map(toindex).values, maxlen=20)
        X_train = train_data[['size', 'avg_price', 'bedrooms', 'bathrooms', 'receptions']]
        y_train = train_data['price']

        words_test = pad_sequences(test_data['text'].map(toindex).values, maxlen=20)
        X_test = test_data[['size', 'avg_price', 'bedrooms', 'bathrooms', 'receptions']]
        y_test = test_data['price']

        # mlp model
        mlp = build_model('mlp', embedding_dim)
        mlp.fit([words_train, X_train], y_train, epochs=10, batch_size=100, verbose=0)
        scores = mlp.evaluate([words_test, X_test], y_test, batch_size=100, verbose=0)
        y_pred = mlp.predict([words_test, X_test])

        # cnn model
        # cnn = build_model('cnn', embedding_dim)
        # cnn.fit([words_train, X_train], y_train, epochs=10, batch_size=100, verbose=0)
        # scores = cnn.evaluate([words_test, X_test], y_test, batch_size=100, verbose=0)
        # y_pred = cnn.predict([words_test, X_test])

        # rnn model
        # rnn = build_model('rnn', embedding_dim)
        # rnn.fit([words_train, X_train], y_train, epochs=10, batch_size=100, verbose=0)
        # scores = rnn.evaluate([words_test, X_test], y_test, batch_size=100, verbose=0)
        # y_pred = rnn.predict([words_test, X_test])

        print("%s: %.2f" % ('mae', scores[0]))
        cvscores.append(scores[0])
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
