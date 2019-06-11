import pickle
import unidecode
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import gensim


model = gensim.models.Word2Vec.load("w2v.model")
word_vectors = model.wv
print(model.most_similar(positive=['disease']))