import pickle
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from timeit import default_timer as timer
import gensim
import numpy as np

f = open('cleaned_notes.pckl', 'rb')
clean_notes = pickle.load(f)
f.close()
start = timer()

notes = []
for note in clean_notes:
        notes.append(word_tokenize(note))

# build the model - look into parameters
model = gensim.models.Word2Vec(notes, size=300, window=10, min_count=2, workers=10)

print('Start training process...') 
model.train(notes,total_examples=len(notes),epochs=10)

embedding_matrix = np.zeros((len(model.wv.vocab), 300))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model.save("w2v.model")
print("Model Saved")


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


end = timer() # around 2-4 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")
