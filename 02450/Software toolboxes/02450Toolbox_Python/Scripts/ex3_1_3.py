# exercise 3.1.4
from sklearn.feature_extraction.text import CountVectorizer

# As before, load the corpus and preprocess:
with open('../Data/textDocs.txt', 'r') as f:
    raw_file = f.read()
corpus = raw_file.split('\n')
corpus = list(filter(None, corpus))

# Load and process the stop words in a similar manner:
with open('../Data/stopWords.txt', 'r') as f:
    raw_file = f.read()
stopwords = raw_file.split('\n')

# When making the CountVectorizer, we now input the stop words:
vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b', 
                             stop_words=stopwords)    
# Determine the terms in the corpus
vectorizer.fit(corpus)
# ... and count the frequency of each term within a document:
X = vectorizer.transform(corpus)
attributeNames = vectorizer.get_feature_names()
N,M = X.shape

# Display the result
print('Document-term matrix analysis (using stop words)')
print()
print('Number of documents (data objects, N):\t %i' % N)
print('Number of terms (attributes, M):\t %i' % M )
print()
print('Found terms (no stop words):')
print(attributeNames)
print()
print('Document-term matrix:')
print(X.toarray())
print()
print('Ran Exercise 3.1.3')