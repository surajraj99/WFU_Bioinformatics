import numpy as np
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax
import tensorflow as tf
import random as rn
import os
import random
import statistics as st
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM, Embedding, Dropout, GRU, Bidirectional
from keras.layers import Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras import regularizers
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

def get_variables_cluster():
    # Reading tokenized notes from panda files
    df = pd.read_hdf('//home//srajendr//PandaFiles//tokenized_notes.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_w2v.h5')
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_GNV.h5')
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('//home//srajendr//PickleFiles//word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('//home//srajendr//PickleFiles//max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading tokenized notes eff from panda files
    df = pd.read_hdf('//home//srajendr//PandaFiles//tokenized_notes_eff.h5')
    notes_eff = df.values.tolist()

    # Reading word2vec word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_w2v_eff.h5')
    embedding_matrix_w2v_eff = df.to_numpy()

    # Reading Google word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('//home//srajendr//PandaFiles//embedding_matrix_GNV_eff.h5')
    embedding_matrix_GNV_eff = df.to_numpy()

    # Reading word index eff from Pickle
    f = open('//home//srajendr//PickleFiles//word_index_eff.pckl', 'rb')
    word_index_eff = pickle.load(f)
    f.close()

    # Reading max length eff from Pickle
    f = open('//home//srajendr//PickleFiles//max_len_eff.pckl', 'rb')
    max_len_eff = pickle.load(f)
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
    df = pd.read_hdf('PandaFiles/tokenized_notes.h5')
    notes = df.values.tolist()

    # Reading word2vec word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_w2v.h5')
    embedding_matrix_w2v = df.to_numpy()

    # Reading Google word embedding matrix from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_GNV.h5')
    embedding_matrix_GNV = df.to_numpy()

    # Reading word index from Pickle
    f = open('PickleFiles/word_index.pckl', 'rb')
    word_index = pickle.load(f)
    f.close()

    # Reading max length from Pickle
    f = open('PickleFiles/max_len.pckl', 'rb')
    max_len = pickle.load(f)
    f.close()

    # Reading tokenized notes eff from panda files
    df = pd.read_hdf('PandaFiles/tokenized_notes_eff.h5')
    notes_eff = df.values.tolist()

    # Reading word2vec word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_w2v_eff.h5')
    embedding_matrix_w2v_eff = df.to_numpy()

    # Reading Google word embedding matrix eff from Panda Files and converting to numpy array
    df = pd.read_hdf('PandaFiles/embedding_matrix_GNV_eff.h5')
    embedding_matrix_GNV_eff = df.to_numpy()

    # Reading word index eff from Pickle
    f = open('PickleFiles/word_index_eff.pckl', 'rb')
    word_index_eff = pickle.load(f)
    f.close()

    # Reading max length eff from Pickle
    f = open('PickleFiles/max_len_eff.pckl', 'rb')
    max_len_eff = pickle.load(f)
    f.close()

    # Reading binary labels
    f = open('PickleFiles/binary_labels.pckl', 'rb')
    binary_labels = pickle.load(f)
    f.close()

    # Reading categorical labels
    f = open('PickleFiles/categorical_labels.pckl', 'rb')
    categorical_labels = pickle.load(f)
    f.close()

    return notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels

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

def make_eff_labels():
    # Binary Labels
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(notes_eff, binary_labels, test_size=0.33, random_state=39)
    X_train_b = np.array(X_train_b)
    X_test_b = np.array(X_test_b)

    # Categorical Labels
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(notes_eff, categorical_labels, test_size=0.33, random_state=39)
    X_train_c = np.array(X_train_c)
    X_test_c = np.array(X_test_c)

    return X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c

#sbatch --no-requeue : command to not repeat

# Choose which setting you are running the model in and comment one othe two next lines out
notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables_cluster()
#notes, embedding_matrix_w2v, embedding_matrix_GNV, word_index, max_len, notes_eff, embedding_matrix_w2v_eff, embedding_matrix_GNV_eff, word_index_eff, max_len_eff, binary_labels, categorical_labels = get_variables_local()

X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_eff_labels()
#X_train_b, X_test_b, y_train_b, y_test_b, X_train_c, X_test_c, y_train_c, y_test_c = make_whole_labels()

X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_train_b, y_train_b, test_size=0.2, random_state=39)

# # temporary for local testing:
# X_train_b = X_train_b[:10]
# y_train_b = y_train_b[:10]
# X_test_b = X_test_b[:][:10]
# y_test_b = y_test_b[:][:10]

# X_train_c = X_train_c[:10]
# y_train_c = y_train_c[:10]
# X_test_c = X_test_c[:][:10]
# y_test_c = y_test_c[:][:10]


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

def evaluate_model(metrics, categorical, model, y_test, X_test):
    y_pred=model.predict(X_test,verbose=1)
    if (categorical): ##Check this out, weird##
        y_pred_coded = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)
        metric=[]
        metric.append(['f1score',f1_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['precision',precision_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['recall',recall_score(y_test,y_pred_coded, average='weighted')])
        metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
        print(metric)
        metrics.append(metric)
    else:
        y_pred_coded=np.where(y_pred>0.5,1,0)
        y_pred_coded=y_pred_coded.flatten()
        metric=[]
        metric.append(['f1score',f1_score(y_test,y_pred_coded)])
        metric.append(['precision',precision_score(y_test,y_pred_coded)])
        metric.append(['recall',recall_score(y_test,y_pred_coded)])
        metric.append(['accuracy',accuracy_score(y_test,y_pred_coded)])
        print(metric)
        metrics.append(metric)
    
    return metrics, y_pred

#################################################################################################

# CNN Model Creation 2
def fit_model(word_index, embedding_matrix, max_len, categorical, X_train, y_train, X_val_b, y_val_b):
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto', restore_best_weights=True) # pateince is number of epochs
    callbacks_list = [earlystop]
    optm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False))
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(Dropout(0.2))
    if (categorical):
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
    model_info=model.fit(X_train, y_train, validation_data=(X_val_b, y_val_b), epochs=10, batch_size=8, callbacks=callbacks_list, verbose=1)

    return model

# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
	# make predictions
	yhats = [model.predict(testX) for model in members]
	yhats = array(yhats)
	# sum across ensemble members
	summed = np.sum(yhats, axis=0)
	# argmax across classes
	result = argmax(summed, axis=1)
	return result
 
# evaluate a specific number of members in an ensemble
def evaluate_n_members(members, n_members, testX, testy):
    # select a subset of members
    subset = members[:n_members]
    # make prediction
    yhat = ensemble_predictions(subset, testX)
    # calculate accuracy
    return accuracy_score(testy, yhat)


# fit all models
n_members = 10
members = [fit_model(word_index_eff, embedding_matrix_GNV_eff, max_len_eff, False, X_train_b, y_train_b, X_val_b, y_val_b) for _ in range(n_members)]
# evaluate different numbers of ensembles on hold out set
single_scores, ensemble_scores = list(), list()
for i in range(1, len(members)+1):
	# evaluate model with i members
	ensemble_score = evaluate_n_members(members, i, X_test_b, y_test_b)
	# evaluate the i'th model standalone
	#testy_enc = to_categorical(y_test)
	_, single_score = members[i-1].evaluate(X_test_b, y_test_b, verbose=0)
	# summarize this step
	print('> %d: single=%.3f, ensemble=%.3f' % (i, single_score, ensemble_score))
	ensemble_scores.append(ensemble_score)
	single_scores.append(single_score)
# summarize average accuracy of a single final model
print('Accuracy %.3f (%.3f)' % (mean(single_scores), std(single_scores)))
# plot score vs number of ensemble members
x_axis = [i for i in range(1, len(members)+1)]
plt.plot(x_axis, single_scores, marker='o', linestyle='None')
plt.plot(x_axis, ensemble_scores, marker='o')
plt.show()