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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import inflect

# function that cleans text
# still need to account for contractions, abbreviations, and numbers/fractions
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english') # or any other list of your choice
def clean_text(text, ):

        # def replace_numbers(text):
        #         words = word_tokenize(text)
        #         p = inflect.engine()
        #         new_words = []
        #         for word in words:
        #                 if word.isdigit():
        #                         new_word = p.number_to_words(word)
        #                         new_words.append(new_word)
        #                 else:
        #                         new_words.append(word)
        #         newWords = " ".join(new_words)
        #         return newWords

        def remove_rare(text):
                tokens = word_tokenize(text)
                freq_dist = nltk.FreqDist(tokens)
                rarewords = list(freq_dist.keys())[-10:]
                new_words = [word for word in tokens if word not in rarewords]
                text = " ".join(new_words)
                return text
                

        def tokenize_text(text):
                return [w for s in sent_tokenize(text) for w in word_tokenize(s)]

        def remove_punctuation_characters(text):
                text = text.translate(str.maketrans('', '', string.punctuation))
                return text

        # def stem_text(text, stemmer=default_stemmer):
        #         tokens = tokenize_text(text)
        #         return ' '.join([stemmer.stem(t) for t in tokens])

        def remove_stopwords(text, stop_words=default_stopwords):
                tokens = [w for w in tokenize_text(text) if w not in stop_words]
                return ' '.join(tokens)
        
        def misc_cleaning(text):
                text = re.sub("-([a-zA-Z]+)", r"\1", text) # replaces hyphen with spaces in case of strings
                text = re.sub(r'[0-9\.]+', '', text) # gets rid of numbers --> may not want this in certain cases
                text = re.sub(' y ', '', text) # gets rid of random y accent stuff scattered through the text
                text = re.sub(' ml ', ' milliliter ', text)
                text = re.sub(' mg ', ' milligram ', text)
                text = re.sub(' kg ', ' kilogram ', text)
                text = re.sub(' yo ', ' year old ', text)
                text = re.sub(' cm ', ' centimeter ', text)
                text = re.sub(' mm ', ' millimeter ', text)
                return text

        text = text.strip(' ') # strip whitespaces
        # text = replace_numbers(text)
        text = misc_cleaning(text) # look at function, random cleaning stuff
        text = text.lower() # lowercase
        # text = stem_text(text) # stemming only works for word vectors that are not pre-trained so don't use in this case
        text = remove_punctuation_characters(text) # remove punctuation and symbols
        text = remove_stopwords(text) # remove stopwords
        text = remove_rare(text) # remove 10 most rare words from each document

        return text

# load all the original unclean notes
f = open('original_notes.pckl', 'rb')
old_notes = pickle.load(f)
f.close()

start = timer()
notes = []
for note in old_notes: # takes 100 - 300 seconds to go through the cleaning for-loop for all notes
        notes.append(clean_text(note))
end = timer()
print(end - start)


# save cleaned notes into a picle file
f = open('cleaned_notes.pckl', 'wb')
pickle.dump(notes, f)
f.close()

print("Done cleaning and saving within " + str(end-start) + " seconds")