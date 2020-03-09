import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, GRU,CuDNNGRU,Input, LSTM, Embedding, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, TimeDistributed, BatchNormalization
from keras.layers import concatenate as lconcat
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


#sess_config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils,plot_model, multi_gpu_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

#### Define number of words, and embedding dimensions
max_words = 34603
embed_dim = 100


def load_embeddings(vec_file):
    print("Loading Glove Model")
    f = open(vec_file, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. {} words loaded!".format(len(model)))
    return model


def tokenize_and_pad(docs, max_words=max_words):
    global t
    t = Tokenizer()
    t.fit_on_texts(docs)
    docs = pad_sequences(sequences=t.texts_to_sequences(docs), maxlen=max_words, padding='post')
    global vocab_size
    vocab_size = len(t.word_index) + 1

    return docs


def oversample(X, docs, y):
    # Get number of rows with imbalanced class
    target = y.sum().idxmax()
    n = y[target].sum()
    # identify imbalanced targets
    imbalanced = y.drop(target, axis=1)
    # For each target, create a dataframe of randomly sampled rows, append to list
    append_list = [y.loc[y[col] == 1].sample(n=n - y[col].sum(), replace=True, random_state=20) for col in
                   imbalanced.columns]
    append_list.append(y)
    y = pd.concat(append_list, axis=0)
    # match y indexes on other inputs
    X = X.loc[y.index]
    docs = pd.DataFrame(docs_train, index=y_train.index).loc[y.index]
    assert (y.index.all() == X.index.all() == docs.index.all())
    return X, docs.values, y


def build_model(output_classes, architecture, aux_shape=aux_shape, vocab_size=vocab_size, embed_dim=embed_dim,
                embedding_matrix=embedding_matrix, max_seq_len=max_words):
    with tf.device('/cpu:0'):
        main_input = Input(shape=(max_seq_len,), name='doc_input')
        main = Embedding(input_dim=vocab_size,
                         output_dim=embed_dim,
                         weights=[embedding_matrix],
                         input_length=max_seq_len,
                         trainable=False)(main_input)

    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        main = Dense(32, activation='relu')(main)
        main = Dropout(0.2)(main)
        main = Flatten()(main)
    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        main = Conv1D(64, 3, strides=1, padding='same', activation='relu')(main)
        # Cuts the size of the output in half, maxing over every 2 inputs
        main = MaxPooling1D(pool_size=3)(main)
        main = Dropout(0.2)(main)
        main = Conv1D(32, 3, strides=1, padding='same', activation='relu')(main)
        main = GlobalMaxPooling1D()(main)
        # model.add(Dense(output_classes, activation='softmax'))
    elif architecture == 'rnn':
        # LSTM network
        main = Bidirectional(CuDNNGRU(64, return_sequences=False), merge_mode='concat')(main)
        main = BatchNormalization()(main)
    elif architecture == "rnn_cnn":
        main = Conv1D(64, 5, padding='same', activation='relu')(main)
        main = MaxPooling1D()(main)
        main = Dropout(0.2)(main)
        main = Bidirectional(CuDNNGRU(32, return_sequences=False), merge_mode='concat')(main)
        main = BatchNormalization()(main)

    else:
        print('Error: Model type not found.')

    auxiliary_input = Input(shape=(aux_shape,), name='aux_input')
    x = lconcat([main, auxiliary_input])
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    main_output = Dense(output_classes, activation='sigmoid', name='main_output')(x)
    model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output], name=architecture)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = multi_gpu_model(model)
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy', auc_roc])

    return model


def plot_metrics(model_dict, metric, x_label, y_label):
    plots = 1
    plt.figure(figsize=[15, 10])
    for model, history in model_dict.items():
        plt.subplot(2, 2, plots)
        plt.plot(history[metric])
        # plt.plot(history.history['val_acc'])
        plt.title('{0} {1}'.format(model, metric))
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plots += 1
    # plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig("Graphs/{}.png".format(metric), format="png")
    plt.show()


def gen():
    print('generator initiated')
    idx = 0
    while True:
        yield [docs_train[:32], X_train[:32]], y_train[:32]
        print('generator yielded a batch %d' % idx)
        idx += 1