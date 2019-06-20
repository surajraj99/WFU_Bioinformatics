import os
import random
import statistics as st
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def get_variables_cluster():
    df = pd.read_hdf('//home//srajendr//WFU//PandaFiles//tokenized_notes.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//WFU//PandaFiles//embedding_matrix_w2v.h5')
    embedding_matrix = df.to_numpy()

    # Reading word index from Pickle
    f = open('//home//srajendr//WFU//PickleFiles//word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('//home//srajendr//WFU//PickleFiles//max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('//home//srajendr//WFU//PickleFiles//binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('//home//srajendr//WFU//PickleFiles//categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix, word_index, max_len, binary_labels, categorical_labels

def get_variables_local():
    df = pd.read_hdf('PandaFiles/tokenized_notes.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_w2v.h5')
    embedding_matrix = df.to_numpy()

    # Reading word index from Pickle
    f = open('PickleFiles/word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('PickleFiles/max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('PickleFiles/binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('PickleFiles/categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix, word_index, max_len, binary_labels, categorical_labels

# Choose which setting you are running the model in and comment one othe two next lines out
notes, embedding_matrix, word_index, max_len, binary_labels, categorical_labels = get_variables_cluster()
#notes, embedding_matrix, word_index, max_len, binary_labels, categorical_labels = get_variables_local()

X_train, X_test, y_train, y_test = train_test_split(notes, binary_labels, test_size=0.33, random_state=39)
X_train = np.array(X_train)
X_test = np.array(X_test)

# create a plot for the model
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    del fig
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

#LSTM Unidirectional model 
def LSTM_Uni(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed):
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
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," ==========================================================================")
        model_info=model.fit(X_train[train], y_train[train], epochs=2, batch_size=2, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
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


# LSTM_Bidirection model
def LSTM_Bidir(embedding_matrix, word_index, max_len):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(100), merge_mode='concat', weights=None))    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    model_infos=[]
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
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


# CNN Model
def CNN_model(embedding_matrix, word_index, max_len):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    callbacks_list = [earlystop]
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    kfold = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X_train,y_train))
    model_infos=[]
    for i,(train, test) in enumerate(kfold):
        print("Fit fold", i+1," **************************************************************************")
        model_info=model.fit(X_train[train], y_train[train], epochs=100, batch_size=64, validation_data=(X_train[test], y_train[test]),
                               callbacks=callbacks_list, verbose=1)
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

seed = 39
y_pred, metric, model_infos = LSTM_Uni(X_train, y_train, X_test, y_test, word_index, embedding_matrix, max_len, seed)
