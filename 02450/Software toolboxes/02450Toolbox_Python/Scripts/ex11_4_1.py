# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors

# load data from Matlab data file
matdata = loadmat('../Data/digits.mat')
X = np.matrix(matdata['X'])
y = np.matrix(matdata['y'])
N, M = np.shape(X)

# Restrict the data to images of "2"
X = X[y.A.ravel()==2,:]
N, M = np.shape(X)



### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1,)

# Plot density estimate of outlier score
figure(1)
bar(range(20),density[:20])
title('Density estimate')

# Plot possible outliers
figure(2)
for k in range(1,21):
    subplot(4,5,k)
    imshow(np.reshape(X[i[k],:], (16,16)).T, cmap=cm.binary)
    xticks([]); yticks([])
    if k==3: title('Gaussian Kernel Density: Possible outliers')



### K-neighbors density estimator
# Neighbor to use:
K = 5
knn = NearestNeighbors(n_neighbors=K+1).fit(X)
def density(X,i):
    '''
    Compute density at observation i in X using LOO.
    Note this code can easily be vectorized for speed
    '''
    D, _ = knn.kneighbors(X[i])
    # don't compute distance to observation itself.
    density = 1. / D[:, 1:].mean(axis=1)
    return density

dens = np.concatenate([density(X,i) for i in range(N)])
# Sort the scores
i = dens.argsort()
dens = dens[i]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(20),dens[:20])
title('KNN density: Outlier score')
# Plot possible outliers
figure(4)
for k in range(1,21):
    subplot(4,5,k)
    imshow(np.reshape(X[i[k],:], (16,16)).T, cmap=cm.binary)
    xticks([]); yticks([])
    if k==3: title('KNN density: Possible outliers')

### K-nearest neigbor average relative density
# Compute the average relative density
def ard(X,i):
    _, J = knn.kneighbors(X[i])
    J = J[0,1:] # don't include i itself.
    return density(X,i) / np.mean( [density(X, j) for j in J] )

avg_rel_density = np.concatenate( [ard(X,i) for i in range(N) ] )

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(20),avg_rel_density[:20])
title('KNN average relative density: Outlier score')
# Plot possible outliers
figure(6)
for k in range(1,21):
    subplot(4,5,k)
    imshow(np.reshape(X[i_avg_rel[k],:], (16,16)).T, cmap=cm.binary)
    xticks([]); yticks([])
    if k==3: title('KNN average relative density: Possible outliers')



### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K+1).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:,K-1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

# Plot k-neighbor estimate of outlier score (distances)
figure(7)
bar(range(20),score[:20])
title('5th neighbor distance: Outlier score')
# Plot possible outliers
figure(8)
for k in range(1,21):
    subplot(4,5,k)
    imshow(np.reshape(X[i[k],:], (16,16)).T, cmap=cm.binary); 
    xticks([]); yticks([])
    if k==3: title('5th neighbor distance: Possible outliers')



# Plot random digits (the first 20 in the data set), for comparison
figure(9)
for k in range(1,21):
    subplot(4,5,k);
    imshow(np.reshape(X[k,:], (16,16)).T, cmap=cm.binary); 
    xticks([]); yticks([])
    if k==3: title('Random digits from data set')    
show()

print('Ran Exercise 11.4.1')