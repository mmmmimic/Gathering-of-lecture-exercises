# exercise 7.4.4
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.preprocessing import OneHotEncoder

from ex7_4_3 import *
np.random.seed(2450)
y = y.squeeze()
0
# Naive Bayes classifier parameters
alpha = 1.0 # pseudo-count, additive parameter (Laplace correction if 1.0 or Lidtstone smoothing otherwise)
fit_prior = True   # uniform prior (change to True to estimate prior from data)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

X = X[:,0:4] # using all 4 letters,
# for using e.g. only third letter or first and last try X[:,[2]] and X[:, [0,3]]

# We need to specify that the data is categorical.
# MultinomialNB does not have this functionality, but we can achieve similar
# results by doing a one-hot-encoding - the intermediate steps in in training
# the classifier are off, but the final result is corrent.
# If we didn't do the converstion MultinomialNB assumes that the numbers are
# e.g. discrete counts of tokens. Without the encoding, the value 26 wouldn't
# mean "the token 'z'", but it would mean 26 counts of some token,
# resulting in 1 and 2 meaning a difference in one count of a given token as
# opposed to the desired 'a' versus 'b'.
X = OneHotEncoder().fit_transform(X=X)

errors = np.zeros(K)
k=0
for train_index, test_index in CV.split(X):
    #print('Crossvalidation fold: {0}/{1}'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    nb_classifier = MultinomialNB(alpha=alpha,
                                  fit_prior=fit_prior)
    nb_classifier.fit(X_train, y_train)
    y_est_prob = nb_classifier.predict_proba(X_test)
    y_est = np.argmax(y_est_prob,1)

    errors[k] = np.sum(y_est!=y_test,dtype=float)/y_test.shape[0]
    k+=1

# Plot the classification error rate
print('Error rate: {0}%'.format(100*np.mean(errors)))

print('Ran Exercise 7.2.4')
