# exercise 9.2.2


import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import dbplot, dbprobplot, bootstrap
from bin_classifier_ensemble import BinClassifierEnsemble
from sklearn.linear_model import LogisticRegression

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth5.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Fit model using bootstrap aggregation (boosting, AdaBoost):

# Number of rounds of bagging
L = 100

# Weights for selecting samples in each bootstrap
weights = np.ones((N,),dtype=float)/N

# Storage of trained log.reg. classifiers fitted in each bootstrap
logits = [0]*L
alpha = np.ones( (L,) )
votes = np.zeros((N,1))
epsi = 0
y_all = np.zeros((N,L))
y = y > 0.5
# For each round of bagging
for l in range(L):
    
    # Extract training set by random sampling with replacement from X and y
    while True : 
        # not a thing of beauty, however log.reg. fails if presented with less than two classes. 
        X_train, y_train = bootstrap(X, y, N, weights) 
        if not (all(y_train==0) or all(y_train == 1)) : break      
    
    # Fit logistic regression model to training data and save result
    # turn off regularization with C. 
    logit_classifier = LogisticRegression(C=1000)

    logit_classifier.fit(X_train, y_train )
    logits[l] = logit_classifier
    y_est = logit_classifier.predict(X).T > 0.5
    
    y_all[:,l] = 1.0 * y_est
    v  = (y_est != y).T
    ErrorRate = np.multiply(weights,v).sum()
    epsi = ErrorRate
    
    alphai = 0.5 * np.log( (1-epsi)/epsi)
    
    weights[y_est == y] = weights[y_est == y] * np.exp( -alphai )
    weights[y_est != y] = weights[y_est != y] * np.exp(  alphai )
    
    weights = weights / sum(weights)
            
    votes = votes + y_est
    alpha[l] = alphai
    print('Error rate: {:2.2f}%'.format(ErrorRate*100))
    
    
# Estimated value of class labels (using 0.5 as threshold) by majority voting
alpha = alpha/sum(alpha)
y_est_ensemble = y_all @ alpha > 0.5

#y_est_ensemble = votes > (L/2)
#y_est_ensemble = mat(y_all) * mat(alpha) - (1-mat(y_all)) * mat(alpha) > 0
ErrorRateEnsemble = sum(y_est_ensemble != y)/N

# Compute error rate
#ErrorRate = (y!=y_est_ensemble).sum(dtype=float)/N
print('Error rate for ensemble classifier: {:.1f}%'.format(ErrorRateEnsemble*100))
 
ce = BinClassifierEnsemble(logits,alpha)
#ce = BinClassifierEnsemble(logits) # What happens if alpha is not included?
plt.figure(1); dbprobplot(ce, X, y, 'auto', resolution=200)
plt.figure(2); dbplot(ce, X, y, 'auto', resolution=200)
#plt.figure(3); plt.plot(alpha);

#%%
plt.figure(4,figsize=(8,8))
for i in range(2):
    plt.plot(X[ (y_est_ensemble==i),0],X[ (y_est_ensemble==i),1],'br'[i] + 'o')

## Incomment the below lines to investigate miss-classifications
#for i in range(2):
#    plt.plot(X[ (y==i),0],X[ (y==i),1],'br'[i] + '.')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')    
plt.show()

print('Ran Exercise 9.2.2')