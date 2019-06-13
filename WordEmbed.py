from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
import numpy as np
import pickle
from timeit import default_timer as timer
from nltk import word_tokenize

f = open('cleaned_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
start = timer()

# another parameter was data --> avoiding that for now
def textTokenize(text):    
    # label=data['resBack'].tolist()
    # label=np.array(label)
    t = Tokenizer()
    t.fit_on_texts(text) #training phase
    word_index = t.word_index #get a map of word index
    sequences = t.texts_to_sequences(text)
    max_len=max(len(sequences))
    print('Found %s unique tokens' % len(word_index))
    text_tok=pad_sequences(sequences, maxlen=max_len)
    return text_tok, word_index, max_len #also return label but avoiding for now


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
    model.save("w2v.model")
    print("Model Saved")


def word_Embed_w2v(word_index, model):   
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

end = timer() # around 13 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")