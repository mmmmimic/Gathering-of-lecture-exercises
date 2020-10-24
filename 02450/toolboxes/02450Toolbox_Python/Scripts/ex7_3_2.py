import sklearn.tree
import sklearn.linear_model

from toolbox_02450 import *
# requires data from exercise 1.5.1
from ex5_1_5 import *

loss = 2
#X,y = X[:,:10], X[:,10:]
# This script crates predictions from three KNN classifiers using cross-validation

K = 10
m = 1
J = 0
r = []
kf = model_selection.KFold(n_splits=K)

for dm in range(m):
    y_true = []
    yhat = []

    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index, :], y[test_index]

        mA = sklearn.tree.DecisionTreeClassifier().fit(X_train, y_train)
        mB = sklearn.tree.DecisionTreeClassifier().fit(X_train, y_train)

        yhatA = mA.predict(X_test)
        yhatB = mB.predict(X_test)[:, np.newaxis]  # justsklearnthings
        y_true.append(y_test)
        yhat.append(np.concatenate([yhatA.reshape(-1,1), yhatB], axis=1))
        r.append(np.sum(yhatA==y_test)/y_test.shape[0] - np.sum(yhatB==y_test)/y_test.shape[0])

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)

if m == 1:
    y_true = np.concatenate(y_true)
    yhat = np.concatenate(yhat)

    alpha = 0.05
    [thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

    print("Theta point estimate", thetahatA, " CI: ", CIA)

    alpha = 0.05
    [thetahatB, CIB] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)

    print("Theta point estimate", thetahatB, " CI: ", CIB)

    print(p_setupII) 
    print(CI_setupII)
