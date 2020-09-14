# exercise 8.3.3 Fit regularized multinomial regression
import matplotlib.pyplot as plt
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import sklearn.linear_model as lm

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth2.mat')
X = mat_data['X']
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]

N, M = X.shape
C = len(classNames)

#%% Model fitting and prediction
# Standardize data based on training set
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit multinomial logistic regression model
regularization_strength = 1e-3
#Try a high strength, e.g. 1e5, especially for synth2, synth3 and synth4
mdl = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', 
                               tol=1e-4, random_state=1, 
                               penalty='l2', C=1/regularization_strength)
mdl.fit(X_train,y_train)
y_test_est = mdl.predict(X_test)

test_error_rate = np.sum(y_test_est!=y_test) / len(y_test)

predict = lambda x: np.argmax(mdl.predict_proba(x),1)
plt.figure(2,figsize=(9,9))
visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
plt.title('LogReg decision boundaries')
plt.show()


# Number of miss-classifications
print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(y_test)))
# %%

plt.figure(2, figsize=(9,9))
plt.hist([y_train, y_test, y_test_est], color=['red','green','blue'], density=True)
plt.legend(['Training labels','Test labels','Estimated test labels'])


print('Ran Exercise 8.3.2')