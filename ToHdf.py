import pandas as pd
import pickle

# Tokenized notes
f = open('PickleFiles/tokenized_notes.pckl', 'rb')
notes = pickle.load(f)
f.close()
df = pd.DataFrame(notes)
df.to_hdf('PandaFiles/tokenized_notes.h5', key='df')



#Emebedding matrix for w2v
f = open('PickleFiles/embedding_matrix_w2v.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('PandaFiles/embedding_matrix_w2v.h5', key='df')



#Embedding matrix for gnv
f = open('PickleFiles/embedding_matrix_GNV.pckl', 'rb')
embedding_matrix = pickle.load(f)
f.close()
df = pd.DataFrame(embedding_matrix)
df.to_hdf('PandaFiles/embedding_matrix_GNV.h5', key='df')
