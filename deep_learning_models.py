import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, CuDNNGRU, Input, Concatenate, LSTM, Embedding, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, TimeDistributed, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec

# Machine Learning Models
# 1. Preprocess dataframe
# 2. Load Word Embeddings
# 3. Train Models
# 4. Model Evaluation
# 5. Continue Training for More Epochs


def build_model(output_classes, architecture):

    # add embedding layer
    input_emb = Input(shape=(1,))
    # input_num = Input(shape=(5,))
    e = Embedding(25, 1, input_length=25, trainable=False)(input_emb)
    flatten = Flatten()(e)
    # conc = Concatenate()([flatten, input_num])

    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        main = Dense(32, activation='relu')(flatten)
        main = Dropout(0.2)(main)

    # elif architecture == 'cnn':
    #     # 1-D Convolutional Neural Network
    #     main = Conv1D(64, 3, strides=1, padding='same', activation='relu')(conc)
    #     # Cuts the size of the output in half, maxing over every 2 inputs
    #     main = MaxPooling1D(pool_size=3)(main)
    #     main = Dropout(0.2)(main)
    #     main = Conv1D(32, 3, strides=1, padding='same', activation='relu')(main)
    #     main = GlobalMaxPooling1D()(main)
    #
    # elif architecture == 'rnn':
    #     # LSTM network
    #     main = Bidirectional(CuDNNGRU(64, return_sequences=False), merge_mode='concat')(conc)
    #     main = BatchNormalization()(main)
    #
    # elif architecture == 'rnn_cnn':
    #     main = Conv1D(64, 5, padding='same', activation='relu')(conc)
    #     main = MaxPooling1D()(main)
    #     main = Dropout(0.2)(main)
    #     main = Bidirectional(CuDNNGRU(32, return_sequences=False), merge_mode='concat')(main)
    #     main = BatchNormalization()(main)
    #
    # else:
    #     print('Error: Model type not found.')

    main_output = Dense(output_classes, activation='sigmoid')(main)
    # model = Model(inputs=[input_emb, input_num], outputs=[main_output], name=architecture)
    model = Model(inputs=input_emb, outputs=[main_output], name=architecture)
    model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def combine_word_vector(words):
    vector = np.array([0] * 25)
    for word in words:
        vector = np.add(wordvectors.word_vec(word), vector)
    return vector


if __name__ == '__main__':

    df = pd.read_csv("data_total_dl.csv")
    input_text = [i for i in df['text']]
    model = Word2Vec(input_text, min_count=1, size=25, workers=3, window=3, sg=0)
    wordvectors = model.wv  # KeyedVectors Instance gets stored
    df['word_vectors'] = df['text'].map(combine_word_vector)

    # Separate into X and Y
    X = df[['size', 'average', 'bedroom-number', 'bathroom-number', 'reception-number']]
    word_vectors = df['word_vectors']
    y = df['price_range']

    # Split into train and test data
    X_train, X_test, y_train, y_test, words_train, words_test = train_test_split(X, y, word_vectors, stratify=y, test_size=0.1, random_state=20) # Save data

    np.save("train_test_data/words_train.npy", words_train)
    np.save("train_test_data//words_test.npy", words_test)

    X_train.to_csv("train_test_data/X_train.csv")
    X_test.to_csv("train_test_data/X_test.csv")

    y_train.to_csv("train_test_data/y_train.csv")
    y_test.to_csv("train_test_data/y_test.csv")

    model = build_model(1, 'mlp')
    print(model.summary())
    model_result = model.fit(words_train, y_train, batch_size=64, epochs=10, verbose=1)