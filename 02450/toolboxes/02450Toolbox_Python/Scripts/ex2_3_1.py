# exercise 2.3.1

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import scipy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt

# Number of principal components to use for classification,
# i.e. the reduced dimensionality
K = [8,10,15,20,30,40,50,60,100,150]

# Load Matlab data file and extract training set and test set
mat_data = loadmat('../Data/zipdata.mat')
X = mat_data['traindata'][:,1:]
y = mat_data['traindata'][:,0]
Xtest = mat_data['testdata'][:,1:]
ytest = mat_data['testdata'][:,0]
N,M = X.shape
Ntest = Xtest.shape[0] # or Xtest[:,0].shape

# Subtract the mean from the data
Y = X - np.ones((N,1))*X.mean(0)
Ytest = Xtest - np.ones((Ntest,1))*X.mean(0)

# Obtain the PCA solution  by calculate the SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)
V = V.T

plt.figure()
rho = (S*S) / (S*S).sum() 
plt.plot(np.cumsum(rho))
# from the figure we can know that when using 40-60 PCA components, the classifier is the best, because 40-60
# PCA components cover most of the information (90%), but not too much, when too much information are given to the classifier,
# it will not be able to classify the tangled data

# Repeat classification for different values of K
error_rates = []
for k in K:
    # Project data onto principal component space,
    Z = Y @ V[:,:k]
    Ztest = Ytest @ V[:,:k]

    # Classify data with knn classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=1)
    knn_classifier.fit(Z,y.ravel())
    y_estimated = knn_classifier.predict(Ztest)

    # Compute classification error rates
    y_estimated = y_estimated.T
    er = (sum(ytest!=y_estimated)/float(len(ytest)))*100
    error_rates.append(er)
    print('K={0}: Error rate: {1:.1f}%'.format(k, er))

# Visualize error rates vs. number of principal components considered
figure()
plot(K,error_rates,'o-')
xlabel('Number of principal components K')
ylabel('Error rate [%]')
show()

print('Ran Exercise 2.3.1')