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
end = timer() # around 2-4 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")