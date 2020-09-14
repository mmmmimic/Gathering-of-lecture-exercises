# exercise 3.1.4
from sklearn.feature_extraction.text import CountVectorizer

# We'll use a widely used stemmer based:
# Porter, M. “An algorithm for suffix stripping.” Program 14.3 (1980): 130-137.
# The stemmer is implemented in the most used natural language processing
# package in Python, "Natural Langauge Toolkit" (NLTK):
from nltk.stem import PorterStemmer

# Load and process the corpus and stop words:
with open('../Data/textDocs.txt', 'r') as f:
    raw_file = f.read()
corpus = raw_file.split('\n')
corpus = list(filter(None, corpus))

with open('../Data/stopWords.txt', 'r') as f:
    raw_file = f.read()
stopwords = raw_file.split('\n')

# To enable stemming when using the sklearn-module, we need to parse an 
# "analyzer" to the vectorizer we've been using. 
# First, we make an object based on the PorterStemmer class, and we also make
# an analyzer object:
stemmer = PorterStemmer()
analyzer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b', 
                           stop_words=stopwords).build_analyzer()
# Using these we'll make a function that can stem words:
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
# ... and finally, we make a vectorizer just like we've done before:
vectorizer = CountVectorizer(analyzer=stemmed_words)    

# Determine the terms:
vectorizer.fit(corpus)
attributeNames = vectorizer.get_feature_names()

# ... and count the occurences:
X = vectorizer.transform(corpus)
N,M = X.shape
X = X.toarray()

# Display the result
print('Document-term matrix analysis (using stop words and stemming)')
print()
print('Number of documents (data objects, N):\t %i' % N)
print('Number of terms (attributes, M):\t %i' % M )
print()
print('Found terms (no stop words, stemmed):')
print(attributeNames)
print()
print('Document-term matrix:')
print(X)
print()
print('Ran Exercise 3.1.4')
print()