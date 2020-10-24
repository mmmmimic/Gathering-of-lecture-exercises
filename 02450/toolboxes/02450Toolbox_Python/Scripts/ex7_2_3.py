from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st
from sklearn.utils import shuffle

# requires data from exercise 1.5.1
from ex5_1_5 import *

X,y = X[:,:10], X[:,10:]
# This script crates predictions from three KNN classifiers using cross-validation
zA = np.ones((1,1))
zB = np.ones((1,1))

kf = model_selection.KFold(10, shuffle=True)
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mA = sklearn.linear_model.LinearRegression().fit(X_train,y_train)
    mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

    yhatA = mA.predict(X_test)
    yhatB = mB.predict(X_test)[:,np.newaxis]  #  justsklearnthings


    # perform statistical comparison of the models
    # compute z with squared error.
    zA = np.vstack((zA, np.abs(y_test - yhatA ) ** 2))
    zB = np.vstack((zB, np.abs(y_test - yhatB ) ** 2))

zA = zA[1:]
zB = zB[1:]

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
print('Confidence interval A: ', CIA)

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis

# compute confidence interval of model B
alpha = 0.05
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))  # Confidence interval
print('Confidence interval B: ', CIB)

z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print('Confidence interval: ', CI)
print('p value: ', p)

# the interval becomes much more narrow, but p-value increases.  