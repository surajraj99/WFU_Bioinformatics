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

# The next few lines of code is the tokenizer from keras. A good way to make the sequences and word_index
MAX_NB_WORDS = 20000 # max number of words in the vocabulary maybe??
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(notes)
sequences = tokenizer.texts_to_sequences(notes) # a bunch of numbers in a list, persumably the index of each word according to something
word_index = tokenizer.word_index # a bunch of numbers and the word associated with each number in a dictionary?

try:
	model = Word2Vec.load("GoogleVectors.model")
except:
	model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
	model.save("GoogleVectors.model")

embedding_matrix = np.zeros((len(model.vocab), 300))
for i in range(len(model.vocab)):
    embedding_vector = model[model.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# may potentially need to turn x train and x test data into these vectors based on word vectors
# function to get each word in each do transformed into a 300 dimensional vector; could compress this with average function later to make it faster
def get_word_vecs(documents):
	notes_vec = []
	for doc in documents:
		doc = []
		for word in doc:
			if word in model.vocab:
				doc.append(model[word])
		notes_vec.append(doc)
		
	return notes_vec 


temp_notes = []
for note in notes:
    temp_notes.append(word_tokenize(note))
notes = temp_notes

# function to get the vector for each document from the words themselves; averaging all the vectors in the document
def document_vector(word2vec_model, doc): # remove out-of-vocabulary words
    t = []
    for word in doc:
        if word in word2vec_model.vocab:
            t.append(word)
    if len(t) > 0:
        return np.mean(word2vec_model[t], axis=0)
    else:
        return []


doc_vecs = []
for doc in notes: #look up each doc in model
	a = document_vector(model, doc)
	if (len(a) > 0):
		doc_vecs.append(a)

print("Total number of documents: " + str(len(doc_vecs)))

end = timer() # around 3-4 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")
