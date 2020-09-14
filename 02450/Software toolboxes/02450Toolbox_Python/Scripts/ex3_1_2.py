# exercise 3.1.4
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Load the textDocs.txt as a long string into raw_file:
with open('../Data/textDocs.txt', 'r') as f:
    raw_file = f.read()
# raw_file contains sentences seperated by newline characters, 
# so we split by '\n':
corpus = raw_file.split('\n')
# corpus is now list of "documents" (sentences), but some of them are empty, 
# because textDocs.txt has a lot of empty lines, we filter/remove them:
corpus = list(filter(None, corpus))

# Display the result
print('Document-term matrix analysis')
print()
print('Corpus (5 documents/sentences):')
print(np.asmatrix(corpus))
print()


# To automatically obtain the bag of words representation, we use sklearn's
# feature_extraction.text module, which has a function CountVectorizer.
# We make a CounterVectorizer:
vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b')   
# The token pattern is a regular expression (marked by the r), which ensures 
# that the vectorizer ignores digit/non-word tokens - in this case, it ensures 
# the 10 in the last document is not recognized as a token. It's not important
# that you should understand it the regexp.

# The object vectorizer can now be used to first 'fit' the vectorizer to the
# corpus, and the subsequently transform the data. We start by fitting:
vectorizer.fit(corpus)
# The vectorizer has now determined the unique terms (or tokens) in the corpus
# and we can extract them using:
attributeNames = vectorizer.get_feature_names()
print('Found terms:')
print(attributeNames)
print()

# The next step is to count how many times each term is found in each document,
# which we do using the transform function:
X = vectorizer.transform(corpus)
N,M = X.shape
print('Number of documents (data objects, N):\t %i' % N)
print('Number of terms (attributes, M):\t %i' % M )
print()
print('Document-term matrix:')
print(X.toarray())
print()
print('Ran Exercise 3.1.2')