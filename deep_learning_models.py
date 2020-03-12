import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, CuDNNGRU, Input, Concatenate, LSTM, Embedding, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, BatchNormalization
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

# Machine Learning Models
# 1. Preprocess dataframe
# 2. Load Word Embeddings
# 3. Train Models
# 4. Model Evaluation
# 5. Continue Training for More Epochs


def build_model(output_classes, architecture, embeddings_matrix, embeddings_dim):

    # add embedding layer
    input_emb = Input(shape=(25,))
    input_num = Input(shape=(5,))
    # words have already been trained
    e = Embedding(input_dim=len(embeddings_matrix),
                  output_dim=embeddings_dim,
                  weights=[embeddings_matrix],
                  input_length=25,
                  trainable=False)(input_emb)
    flatten = Flatten()(e)
    conc = Concatenate()([flatten, input_num])

    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        main = Dense(32, activation='relu')(conc)
        main = Dropout(0.1)(main) # 0.2

    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        main = Conv1D(32, 3, strides=1, padding='same', activation='relu')(conc)
        # Cuts the size of the output in half, maxing over every 2 inputs
        main = MaxPooling1D(pool_size=3)(main)
        main = Dropout(0.2)(main)
        main = Conv1D(32, 3, strides=1, padding='same', activation='relu')(main)
        main = GlobalMaxPooling1D()(main)

    elif architecture == 'rnn':
        # LSTM network
        main = Bidirectional(CuDNNGRU(64, return_sequences=False), merge_mode='concat')(conc)
        main = BatchNormalization()(main)

    elif architecture == 'rnn_cnn':
        main = Conv1D(32, 5, padding='same', activation='relu')(conc)
        main = MaxPooling1D()(main)
        main = Dropout(0.2)(main)
        main = Bidirectional(CuDNNGRU(32, return_sequences=False), merge_mode='concat')(main)
        main = BatchNormalization()(main)

    else:
        print('Error: Model type not found.')

    main_output = Dense(output_classes, activation='sigmoid')(main)
    model = Model(inputs=[input_emb, input_num], outputs=[main_output], name=architecture)

    # model = Model(inputs=input_emb, outputs=[main_output], name=architecture)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

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
    myPath = './key_features_word2vec'
    Word2VecModel = Word2Vec.load(myPath)
    df = pd.read_csv("data.csv")
    input_text = [i for i in df['text']]
    model = Word2Vec(input_text, min_count=1, size=25, workers=3, window=3, sg=0)
    wordvectors = model.wv  # KeyedVectors Instance gets stored

    vocab_list = [word for word, Vocab in Word2VecModel.wv.vocab.items()]
    word_index = {" ": 0}
    word_vector = {}
    embedding_dim = Word2VecModel.vector_size
    embeddings_matrix = np.zeros((len(vocab_list) + 1, embedding_dim))

    for i in range(len(vocab_list)):
        word = vocab_list[i]
        word_index[word] = i+1
        word_vector[word] = Word2VecModel.wv[word]
        embeddings_matrix[i + 1] = Word2VecModel.wv[word]

    X_temp = df['text'].map(toindex)
    X_words = pad_sequences(X_temp.values, maxlen=25)

    # Separate into X and Y
    X = df[['size', 'average', 'bedroom-number', 'bathroom-number', 'reception-number']]
    y = df['class']
    y = to_categorical(y)
    # # Split into train and test data
    X_train, X_test, y_train, y_test, words_train, words_test = train_test_split(X, y, X_words, stratify=y, test_size=0.1, random_state=20) # Save data

    np.save("train_test_data/words_train.npy", words_train)
    np.save("train_test_data//words_test.npy", words_test)

    X_train.to_csv("train_test_data/X_train.csv")
    X_test.to_csv("train_test_data/X_test.csv")

    mlp = build_model(8, 'mlp', embeddings_matrix, embedding_dim)
    print(mlp.summary())
    model_result = mlp.fit([words_train, X_train], y_train, batch_size=32, epochs=10, verbose=1)

    print(mlp.evaluate([words_test, X_test], y_test, batch_size=64))