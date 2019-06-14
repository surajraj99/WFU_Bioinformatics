from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.models import Model
from keras import metrics
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

f = open('Pickle Files\\tokenized_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()

f = open('Pickle Files\\embedding_matrix_w2v.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()

f = open('Pickle Files\\word_index.pckl', 'rb')
word_index = pickle.load(f)
f.close()

labels = np.zeros(len(notes))
indices = np.random.choice(np.arange(labels.size), replace=False, size=int(labels.size * 0.5))
labels[indices] = 1


X_train, X_test, y_train, y_test = train_test_split(notes, labels, test_size=0.33, random_state=42)

#define the LSTM baseline model for parameter tuning
def LSTM_baseline(word_index, embedding_matrix, max_len):
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix],input_length=max_len,trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = LSTM_baseline(word_index, embedding_matrix, 263887)

print(model.metrics_names)
_ = model.fit(X_train, y_train, batch_size = 32, epochs = 2, verbose = 1, validation_split = 0.1)
score = model.evaluate(X_test, y_test, batch_size = 32, verbose = 1)
