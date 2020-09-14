# exercise 3.1.5
import numpy as np
import scipy.linalg as linalg
from similarity import similarity

from ex3_1_4 import *

# Query vector
q = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
# notice, that you could get the query vector using the vectorizer, too:
#q = vectorizer.transform(['matrix rank solv'])
#q = np.asarray(q.toarray())
# or use any other string:
#q = vectorizer.transform(['Can I Google how to fix my problem?'])
#q = np.asarray(q.toarray())

# Method 1 ('for' loop - slow)
N = np.shape(X)[0]; # get the number of data objects
sim = np.zeros((N,1)) # allocate a vector for the similarity
for i in range(N):
    x = X[i,:] # Get the i'th data object (here: document)
    sim[i] = q/linalg.norm(q) @ x.T/linalg.norm(x) # Compute cosine similarity

# Method 2 (one line of code with no iterations - faster)
sim = (q @ X.T).T / (np.sqrt(np.power(X,2).sum(axis=1)) * np.sqrt(np.power(q,2).sum()))

# Method 3 (use the "similarity" function)
sim = similarity(X, q, 'cos');


# Display the result
print('Query vector:\n {0}\n'.format(q))
print('Similarity results:\n {0}'.format(sim))

print('Ran Exercise 3.1.5')