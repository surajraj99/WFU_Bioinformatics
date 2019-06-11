from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from gensim import models
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pickle
from timeit import default_timer as timer

f = open('cleaned_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
start = timer()

tagged_notes = []
tag = 0
for note in notes:
    tagged_notes.append(TaggedDocument(note, tags=['0' + str(tag)]))
    tag += 1

model = gensim.models.Doc2Vec(vector_size = 300, windows = 5, min_count = 5, workers = 4, epochs = 20)
model.build_vocab(tagged_notes)

print('Start training process...') 
model.train(tagged_notes, total_examples=model.corpus_count, epochs=model.epochs)
 
model.save("Doc2Vec.model")
print("Model Saved")

# need to sepcifically get only vectors and not tags
docvecs = model.docvecs.vectors_docs

end = timer()

print("Time taken: " + str(end-start))
print(len(docvecs))
