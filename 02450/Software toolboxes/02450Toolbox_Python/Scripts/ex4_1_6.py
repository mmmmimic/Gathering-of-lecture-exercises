# exercise 4.1.6

from matplotlib.pyplot import (figure, subplot, title, imshow, xticks, yticks, 
                               show, cm)
import scipy.linalg as linalg
from scipy.io import loadmat
import numpy as np

# Digits to include in analysis (to include all: n = range(10))
n = [0]

# Load Matlab data file to python dict structure
# and extract variables of interest
traindata = loadmat('../Data/zipdata.mat')['traindata']
X = traindata[:,1:]
y = traindata[:,0]
N, M = X.shape
C = len(n)

# Remove digits that are not to be inspected
class_mask = np.zeros(N).astype(bool)
for v in n:
    cmsk = (y==v)
    class_mask = class_mask | cmsk
X = X[class_mask,:]
y = y[class_mask]
N = np.shape(X)[0]

mu = X.mean(axis=0)
s = X.std(ddof=1, axis=0)
S = np.cov(X, rowvar=0, ddof=1)

figure()
subplot(1,2,1)
I = np.reshape(mu, (16,16))
imshow(I, cmap=cm.gray_r)
title('Mean')
xticks([]); yticks([])
subplot(1,2,2)
I = np.reshape(s, (16,16))
imshow(I, cmap=cm.gray_r)
title('Standard deviation')
xticks([]); yticks([])

show()

print('Ran Exercise 4.1.6')