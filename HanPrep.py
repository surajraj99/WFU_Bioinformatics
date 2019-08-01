import numpy as np
import pandas as pd
import pickle
import sys
import os
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim.models import Word2Vec

import nltk
import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import os
import csv
import pickle
from timeit import default_timer as timer
import inflect
from autocorrect import spell
from collections import OrderedDict
import progressbar as pb

# function that cleans text
# still need to account for contractions, abbreviations, and numbers/fractions
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your choice
def clean_text(i, text, notes_concepts, replace_numbers = False, remove_rare = False, remove_punctuation = False, stem_text = False, remove_stopwords = False, remove_num = False , spell_check = False, remove_repeat = False):
        def misc_cleaning(text):
                text = re.sub("-([a-zA-Z]+)", r"\1", text) # replaces hyphen with spaces in case of strings
                text = re.sub(' y ', '', text) # gets rid of random y accent stuff scattered through the text
                text = re.sub('yyy', 'y', text)
                text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
                text = re.sub(r"what's", "what is ", text)
                text = re.sub(r"\'s", " ", text)
                text = re.sub(r"\'ve", " have ", text)
                text = re.sub(r"can't", "cannot ", text)
                text = re.sub(r"n't", " not ", text)
                text = re.sub(r"i'm", "i am ", text)
                text = re.sub(r"\'re", " are ", text)
                text = re.sub(r"\'d", " would ", text)
                text = re.sub(r"\'ll", " will ", text)
                text = re.sub(r",", " ", text)
                text = re.sub(r"\.", " ", text)
                text = re.sub(r"!", " ! ", text)
                text = re.sub(r"\/", " ", text)
                text = re.sub(r"\^", " ^ ", text)
                text = re.sub(r"\+", " + ", text)
                text = re.sub(r"\-", " - ", text)
                text = re.sub(r"\=", " = ", text)
                text = re.sub(r"'", " ", text)
                text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
                text = re.sub(r":", " : ", text)
                text = re.sub(r" e g ", " eg ", text)
                text = re.sub(r" b g ", " bg ", text)
                text = re.sub(r" u s ", " american ", text)
                text = re.sub(r"\0s", "0", text)
                text = re.sub(r" 9 11 ", "911", text)
                text = re.sub(r"e - mail", "email", text)
                text = re.sub(r"j k", "jk", text)
                text = re.sub(r"\s{2,}", " ", text)
                return text

        # function to tokenize text which is used in a lot of the later processing
        def tokenize_text(text):
                return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

        text = text.strip(' ') # strip whitespaces
        text = misc_cleaning(text) # look at function, random cleaning stuff
        text = text.lower() # lowercase

        if remove_repeat:
                sentences = sent_tokenize(text)
                sentences = list(dict.fromkeys(sentences))
                text = " ".join(sentences)
        
        # removes punctuation
        if remove_punctuation:
                text = text.translate(str.maketrans('', '', string.punctuation))

        # does a spellcheck of the text and corrects words that are misspelled if they're least frequent
        if spell_check:
                concepts = notes_concepts[i]
                tokens = word_tokenize(text)
                freq_dist = nltk.FreqDist(tokens)
                if (len((list(freq_dist.keys())[-1:])) > 0):
                        testword = (list(freq_dist.keys())[-1:])[0]
                        lowest_frequency = freq_dist.freq(testword)
                        rarewords = []
                        for word in tokens:
                                if freq_dist.freq(word) == lowest_frequency:
                                        rarewords.append(word)
                        rarewords[:] = [word for word in rarewords if word not in concepts]
                        correctwords = []
                        for word in rarewords:
                                correctwords.append(spell(word))
                for i in range(len(tokens)):
                        for a in range(len(rarewords)):
                                if tokens[i] == rarewords[a]:
                                        tokens[i] = correctwords[a]
                text = " ".join(tokens)

        # optional: replaces numbers ("3") with their word counterparts ("three")
        if replace_numbers:
                words = word_tokenize(text)
                p = inflect.engine()
                new_words = []
                for word in words:
                        if word.isdigit():
                                new_word = p.number_to_words(word)
                                new_words.append(new_word)
                        else:
                                new_words.append(word)
                text = " ".join(new_words)

        # optional: removes the rarest words in each text --> right now it's 10
        if remove_rare:
                tokens = word_tokenize(text)
                freq_dist = nltk.FreqDist(tokens)
                rarewords = list(freq_dist.keys())[-10:]
                new_words = [word for word in tokens if word not in rarewords]
                text = " ".join(new_words)

        # optional: stems text using Porter Stemmer
        if stem_text:
                stemmer = default_stemmer
                tokens = tokenize_text(text)
                text = " ".join([stemmer.stem(t) for t in tokens])

        # removes stop words such as "a", "the", etc.
        if remove_stopwords:
                stop_words = default_stopwords
                tokens = [w for w in tokenize_text(text) if w not in stop_words]
                text = " ".join(tokens)
        
        # optional: removes numbers completely from the ext
        if remove_num:
                text=text.split()
                text=[x for x in text if not x.isnumeric()]
                text= " ".join(text)

        return text

def lengths(x):
    length=[]
    for t in x:
        length.append(len(t))
    return length

# load all the original unclean notes
def load_stuff_cluster():
        f = open('//home//srajendr//PickleFiles//original_notes.pckl', 'rb')
        s = pickle.load(f)
        f.close()

        # load all the concepts
        f = open('//home//srajendr//PickleFiles//notes_concepts.pckl', 'rb')
        notes_concepts = pickle.load(f)
        f.close()

        return s, notes_concepts

def load_stuff_local():
        f = open('PickleFiles//original_notes.pckl', 'rb')
        s = pickle.load(f)
        f.close()

        # load all the concepts
        f = open('PickleFiles//notes_concepts.pckl', 'rb')
        notes_concepts = pickle.load(f)
        f.close()

        return s, notes_concepts

s, notes_concepts = load_stuff_local()
notes = []
patient = []
joined_notes = []
n = []

for i, note in enumerate(s): # takes --- seconds to go through the cleaning for-loop for all notes
        sentences = sent_tokenize(note)
        for j, sentence in enumerate(sentences):
            patient.append(clean_text(i, sentence, notes_concepts, remove_punctuation = True, remove_stopwords = True, remove_repeat = True, spell_check=True))
        notes.append(patient)
        joined_notes.append(" ".join(patient))
        n.append(joined_notes)
        joined_notes = []
        patient = []

def make_data_index(notes):
    MAX_SENTS = max(lengths(notes))
    all_words = []
    MAX_SENT_LENGTH = 0
    for note in notes:
        for sent in note:
            words = word_tokenize(sent)
            all_words.append(words)
            if len(words) > MAX_SENT_LENGTH:
                MAX_SENT_LENGTH = len(words)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_words)
    word_index = tokenizer.word_index

    data = np.zeros((len(notes), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(notes):
        for j, sent in enumerate(sentences):
            if j< MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    if k<MAX_SENT_LENGTH:
                        data[i,j,k] = tokenizer.word_index[word]
                        k=k+1

    return word_index, data, MAX_SENTS, MAX_SENT_LENGTH

def word_Embed_GNV(word_index):   
    pretrain = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.wv.vocab)):
        word=pretrain.wv.index2word[i]
        if word is not None:
            embedding_index[word]=pretrain.wv[word]  
    #extract word embedding for train and test data
    vocab_size=len(word_index)+1
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
    	embedding_vector = embedding_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

def make_w2v_model(notes, window, workers, epochs):
    model = gensim.models.Word2Vec(notes, size=300, window=window, min_count=2, workers=workers)
    print('Start training process...') 
    model.train(notes,total_examples=len(notes),epochs=epochs)
    model.save("w2vhan.model")
    print("Model Saved")


def word_Embed_w2v(word_index, model):   
    pretrain = model
    #convert pretrained word embedding to a dictionary
    embedding_index=dict()
    for i in range(len(pretrain.wv.vocab)):
        word=pretrain.wv.index2word[i]
        if word is not None:
            embedding_index[word]=pretrain.wv[word]  
    #extract word embedding for train and test data
    vocab_size=len(word_index)+1
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
    	embedding_vector = embedding_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    return embedding_matrix

word_index, data, MAX_SENTS, MAX_SENT_LENGTH = make_data_index(notes)
embedding_matrix_GNV = word_Embed_GNV(word_index)
make_w2v_model(notes, 5, 10, 20)
w2v_model = Word2Vec.load("w2vhan.model")
embedding_matrix_w2v = word_Embed_w2v(word_index, w2v_model)

def pickle_whole_variable():
    f = open('//home//srajendr//PickleFiles//han_data.pckl', 'wb')
    pickle.dump(data, f)
    f.close()
    print("Saved Tokenized Data")

    f = open('//home//srajendr//PickleFiles//embedding_matrix_GNV_HAN.pckl', 'wb')
    pickle.dump(embedding_matrix_GNV, f)
    f.close()
    print("Saved Google Vector Word Embedding Matrix for HAN")

    f = open('//home//srajendr//PickleFiles//embedding_matrix_w2v_HAN.pckl', 'wb')
    pickle.dump(embedding_matrix_w2v, f)
    f.close()
    print("Saved Word 2 Vector Embedding Matrix for HAN")

    f = open('//home//srajendr//PickleFiles//word_index_han.pckl', 'wb')
    pickle.dump(word_index, f)
    f.close()
    print("Saved Word Indices for HAN")

    f = open('//home//srajendr//PickleFiles//max_sents.pckl', 'wb')
    pickle.dump(MAX_SENTS, f)
    f.close()
    print("Saved Maximum Length of Notes")

    f = open('//home//srajendr//PickleFiles//max_sent_len.pckl', 'wb')
    pickle.dump(MAX_SENT_LENGTH, f)
    f.close()
    print("Saved Maximum Length of Notes")

def local_pickle_whole_variable():
    f = open('PickleFiles//han_data.pckl', 'wb')
    pickle.dump(data, f)
    f.close()
    print("Saved Tokenized Data")

    f = open('PickleFiles//embedding_matrix_GNV_HAN.pckl', 'wb')
    pickle.dump(embedding_matrix_GNV, f)
    f.close()
    print("Saved Google Vector Word Embedding Matrix for HAN")

    f = open('PickleFiles//embedding_matrix_w2v_HAN.pckl', 'wb')
    pickle.dump(embedding_matrix_w2v, f)
    f.close()
    print("Saved Word 2 Vector Embedding Matrix for HAN")

    f = open('PickleFiles//word_index_han.pckl', 'wb')
    pickle.dump(word_index, f)
    f.close()
    print("Saved Word Indices for HAN")

    f = open('PickleFiles//max_sents.pckl', 'wb')
    pickle.dump(MAX_SENTS, f)
    f.close()
    print("Saved Maximum Length of Notes")

    f = open('PickleFiles//max_sent_len.pckl', 'wb')
    pickle.dump(MAX_SENT_LENGTH, f)
    f.close()
    print("Saved Maximum Length of Notes")

local_pickle_whole_variable()
print("Done")