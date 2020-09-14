# exercise 4.1.7

from matplotlib.pyplot import (figure, subplot, imshow, xticks, yticks, title,
                               cm, show)
import numpy as np
from scipy.io import loadmat

# Digits to include in analysis (to include all, n = range(10) )
n = [1]

# Number of digits to generate from normal distributions
ngen = 10

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat('../Data/zipdata.mat')['traindata']
X = traindata[:,1:]
y = traindata[:,0]
N, M = np.shape(X) #or X.shape
C = len(n)

# Remove digits that are not to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = (y==v)
    class_mask = class_mask | cmsk
X = X[class_mask,:]
y = y[class_mask]
N = np.shape(X)[0] # or X.shape[0]

mu = X.mean(axis=0)
s = X.std(ddof=1, axis=0)
S = np.cov(X, rowvar=0, ddof=1)

# Generate 10 samples from 1-D normal distribution
Xgen = np.random.randn(ngen,256)
for i in range(ngen):
    Xgen[i] = np.multiply(Xgen[i],s) + mu

# Plot images
figure()
for k in range(ngen):
    subplot(2, np.ceil(ngen/2.), k+1)
    I = np.reshape(Xgen[k,:], (16,16))
    imshow(I, cmap=cm.gray_r);
    xticks([]); yticks([])
    if k==1: title('Digits: 1-D Normal')


# Generate 10 samples from multivariate normal distribution
Xmvgen = np.random.multivariate_normal(mu, S, ngen)
# Note if you are investigating a single class, then you may get: 
# """RuntimeWarning: covariance is not positive-semidefinite."""
# Which in general is troublesome, but here is due to numerical imprecission


# Plot images
figure()
for k in range(ngen):
    subplot(2, np.ceil(ngen/2.), k+1)
    I = np.reshape(Xmvgen[k,:], (16,16))
    imshow(I, cmap=cm.gray_r);
    xticks([]); yticks([])
    if k==1: title('Digits: Multivariate Normal')

show()

print('Ran Exercise 4.1.7')