# exercise 9.1.1

from matplotlib.pyplot import figure, show
#import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine2.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'][0]]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# K-fold crossvalidation with stratified folds
K = 2
CV = StratifiedKFold(K, shuffle=True)

k=0
for train_index, test_index in CV.split(X,y):
    print(train_index)
    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    logit_classifier = LogisticRegression()
    logit_classifier.fit(X_train, y_train)

    y_test_est = logit_classifier.predict(X_test).T
    p = logit_classifier.predict_proba(X_test)[:,1].T

    figure(k)
    rocplot(p, y_test)

    figure(k+1)
    confmatplot(y_test,y_test_est)

    k+=2
    
show()    

print('Ran Exercise 9.1.1')