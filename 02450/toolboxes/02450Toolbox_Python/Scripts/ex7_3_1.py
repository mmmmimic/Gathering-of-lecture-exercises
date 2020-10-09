import sklearn.tree
import sklearn.linear_model

from toolbox_02450 import *
# requires data from exercise 1.5.1
from ex5_1_5 import *

loss = 2
X,y = X[:,:10], X[:,10:]
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

        mA = sklearn.linear_model.LinearRegression().fit(X_train, y_train)
        mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

        yhatA = mA.predict(X_test)
        yhatB = mB.predict(X_test)[:, np.newaxis]  # justsklearnthings
        y_true.append(y_test)
        yhat.append( np.concatenate([yhatA, yhatB], axis=1) )

        r.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )

# Initialize parameters and run test appropriate for setup II
alpha = 0.05
rho = 1/K
p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)

if m == 1:
    y_true = np.concatenate(y_true)[:,0]
    yhat = np.concatenate(yhat)

    # note our usual setup I ttest only makes sense if m=1.
    zA = np.abs(y_true - yhat[:,0] ) ** loss
    zB = np.abs(y_true - yhat[:,1] ) ** loss
    z = zA - zB

    CI_setupI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p_setupI = st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value

    print( [p_setupII, p_setupI] )
    print(CI_setupII, CI_setupI )
