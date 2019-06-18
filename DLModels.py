import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import metrics
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping

# Reading tokenized notes from Panda Files
df = pd.read_hdf('PandaFiles/tokenized_notes.h5')
notes = df.values.tolist()

# Reading word2vec word embedding matrix from Panda Files and converting to numpy array
df = pd.read_hdf('PandaFiles/embedding_matrix_w2v.h5')
embedding_matrix = df.to_numpy()

# Reading word index from Panda Files
f = open('PickleFiles/word_index.pckl', 'rb')
word_index = pickle.load(f)
f.close()

# f = open('PickleFiles\\max_len.pckl', 'rb')
# max_len = pickle.load(f)
# f.close()
max_len = 263887 # get rid of later

labels = np.zeros(len(notes))
indices = np.random.choice(np.arange(labels.size), replace=False, size=int(labels.size * 0.5))
labels[indices] = 1

X_train, X_test, y_train, y_test = train_test_split(notes, labels, test_size=0.33, random_state=39)
X_train = np.array(X_train)
X_test = np.array(X_test)

#define the LSTM baseline model for parameter tuning
def LSTM_baseline(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train, y_train))
    model_infos = []
    for i, (train, test) in enumerate(kfold):
        print("Fit fold", i+1, " ***********************************")
        model_info = model.fit(X_train[train], y_train[train], batch_size = 32, epochs = 2, verbose = 1, 
                validation_data=(X_train[test], y_train[test]))
        print("Performance plot of fold {}:".format(i+1))
        plot_model_history(model_info)
        model_infos.append(model_info)
    
    # Final evaluation of the model
    y_pred=model.predict(X_test,verbose=1)
    y_pred_coded=np.where(y_pred>0.5,1,0)
    y_pred_coded=y_pred_coded.flatten()
    
    metric=[]
    metric.append(['f1score',f1_score(y_test,y_pred_coded)])
    metric.append(['precision',precision_score(y_test,y_pred_coded)])
    metric.append(['recall',recall_score(y_test,y_pred_coded)])
    metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])

    return y_pred, metric, model_infos
        

def GRU_model(embedding_matrix, word_index, max_len):
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def LSTM_Bidir(embedding_matrix, word_index, max_len):
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100), merge_mode='concat', weights=None))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def CNN_model(embedding_matrix, word_index, max_len):
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

seed = 39
y_pred, metric, model_infos = LSTM_baseline(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed)
