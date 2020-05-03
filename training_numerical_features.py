import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers import Flatten, Reshape, Conv1D, MaxPooling1D, SimpleRNN
from keras import optimizers
import matplotlib.pyplot as plt


def build_model(architecture):

    # add embedding layer
    input_num = Input(shape=(5, ))

    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        main = Dense(100, activation='relu')(input_num)
        main = Dropout(0.02)(main)
        main = Dense(100, activation='relu')(main)
        main = Dropout(0.02)(main)

    elif architecture == 'cnn':
        main = Reshape((5, 1))(input_num)
        main = Conv1D(50, 3, strides=1, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Dropout(0.02)(main)
        main = Conv1D(50, 3, strides=1, padding='same', activation='relu')(main)
        main = MaxPooling1D(pool_size=2)(main)
        main = Dropout(0.02)(main)
        main = Flatten()(main)

    elif architecture == 'rnn':
        main = Reshape((5, 1))(input_num)
        # simpleRNN
        main = SimpleRNN(50)(main)
        main = Dropout(0.02)(main)

    else:
        print('Error: Model type not found.')

    main_output = Dense(1)(main)
    model = Model(inputs=[input_num], outputs=[main_output], name=architecture)
    sgd = optimizers.SGD(lr=0.1)
    model.compile(optimizer=sgd, loss='mean_absolute_error', metrics=['mae'])

    return model


if __name__ == '__main__':

    cvscores = []

    for fold in range(1, 11):
        train_data = pd.read_pickle("./train_test_data/train_kfold" + str(fold) + ".pkl")
        test_data = pd.read_pickle("./train_test_data/test_kfold" + str(fold) + ".pkl")

        X_train = train_data[['size', 'avg_price', 'bedrooms', 'bathrooms', 'receptions']]
        y_train = train_data['price']

        X_test = test_data[['size', 'avg_price', 'bedrooms', 'bathrooms', 'receptions']]
        y_test = test_data['price']

        # mlp model
        mlp = build_model('mlp')
        mlp.fit(X_train, y_train, epochs=10, batch_size=100, verbose=0)
        scores = mlp.evaluate(X_test, y_test, batch_size=100, verbose=0)
        y_pred = mlp.predict(X_test)

        # cnn model
        # cnn = build_model('cnn')
        # cnn.fit(X_train, y_train, epochs=10, batch_size=100, verbose=0)
        # scores = cnn.evaluate(X_test, y_test, batch_size=100, verbose=0)
        # y_pred = cnn.predict( X_test)

        # rnn model
        # rnn = build_model('rnn')
        # rnn.fit(X_train, y_train, epochs=10, batch_size=100, verbose=0)
        # scores = rnn.evaluate(X_test, y_test, batch_size=100, verbose=0)
        # y_pred = rnn.predict(X_test)

        print("%s: %.2f" % ('mae', scores[0]))
        cvscores.append(scores[0])
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
