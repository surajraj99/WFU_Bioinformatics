from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
import numpy as np
import pickle
from timeit import default_timer as timer

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

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

embedding_matrix = np.zeros((len(model.vocab), 300))
for i in range(len(model.vocab)):
    embedding_vector = model[model.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

end = timer() # around 3-4 minutes to run the whole thing
print("Done within " + str(end-start) + " seconds")