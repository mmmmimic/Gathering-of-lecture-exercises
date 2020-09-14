# exercise 10_1_5
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import k_means


# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wildfaces.mat')
#mat_data = loadmat('../Data/digits.mat') #<-- uncomment this for using the digits dataset 

X = mat_data['X']
N, M = X.shape
# Image resolution and number of colors
x = 40 #<-- change this for using the digits dataset
y = 40 #<-- change this for using the digits dataset
c = 3 #<-- change this for using the digits dataset


# Number of clusters:
K = 10

# Number of repetitions with different initial centroid seeds
S = 1

# Run k-means clustering:
centroids, cls, inertia = k_means(X, K, verbose=True, max_iter=100, n_init=S)


# Plot results:

# Plot centroids
plt.figure(1)
n1 = np.ceil(np.sqrt(K/2)); n2 = np.ceil(np.float(K)/n1)

#For black and white, cmap=plt.cm.binary, else default
cmap = plt.cm.binary if c==1 else None 

for k in range(K):
    plt.subplot(n1,n2,k+1)
    # Reshape centroids to fit resolution and colors
    img = np.reshape(centroids[k,:],(c,x,y)).T
    if c == 1: # if color is single-color/gray scale
        # Squeeze out singleton dimension
        # and flip the image (cancel out previos transpose)
        img = np.squeeze(img).T
    plt.imshow(img,interpolation='None', cmap=cmap)
    plt.xticks([]); plt.yticks([])
    if k==np.floor((n2-1)/2): plt.title('Centroids')

# Plot few randomly selected faces and their nearest centroids    
L = 5       # number of images to plot
j = np.random.randint(0, N, L)
plt.figure(2)
for l in range(L):
    plt.subplot(2,L,l+1)
    img = np.resize(X[j[l],:],(c,x,y)).T
    if c == 1:
        img = np.squeeze(img).T
    plt.imshow(img,interpolation='None', cmap=cmap)
    plt.xticks([]); plt.yticks([])
    if l==np.floor((L-1)/2): plt.title('Randomly selected faces and their centroids')
    plt.subplot(2,L,L+l+1)
    img = np.resize(centroids[cls[j[l]],:],(c,x,y)).T
    if c == 1:
        img = np.squeeze(img).T
    plt.imshow(img,interpolation='None', cmap=cmap)
    plt.xticks([]); plt.yticks([])

plt.show()

print('Ran Exercise 10.1.5')