import os
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer, InputSpec


def make_whole_labels():
    # Binary Labels
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(notes, binary_labels, test_size=0.33, random_state=39)
    X_train_b = np.array(X_train_b)
    X_test_b = np.array(X_test_b)

    # Categorical Labels
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(notes, categorical_labels, test_size=0.33, random_state=39)
    X_train_c = np.array(X_train_c)
    X_test_c = np.array(X_test_c)

    return X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c

def get_variables_cluster():
    # Reading tokenized notes from panda files
    df = pd.read_hdf('//home//srajendr//PandaFiles//han_data.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_w2v_han.h5')
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_GNV_han.h5')
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('//home//srajendr//PickleFiles//max_sents.pckl', 'rb')
    MAX_SENTS = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('//home//srajendr//PickleFiles//max_sent_len.pckl', 'rb')
    MAX_SENT_LENGTH = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('//home//srajendr//PickleFiles//binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('//home//srajendr//PickleFiles//categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels

def get_variables_local():
    # Reading tokenized notes from panda files
    df = pd.read_hdf('PandaFiles/han_data.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_w2v_han.h5')
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_GNV_han.h5')
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('PickleFiles/max_sents.pckl', 'rb')
    MAX_SENTS = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('PickleFiles/max_sent_len.pckl', 'rb')
    MAX_SENT_LENGTH = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('PickleFiles/binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('PickleFiles/categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
preds = Dense(len(macronum), activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("Hierachical LSTM")
model.summary()

cp=ModelCheckpoint('model_han_.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
history=model.fit(x_train_b, y_train_b, validation_data=(x_val, y_val),
          epochs=15, batch_size=2,callbacks=[cp])