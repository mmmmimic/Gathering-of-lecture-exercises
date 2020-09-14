# exercise 10.1.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.cluster import k_means

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth1.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# Number of clusters:
K = 4

# K-means clustering:
centroids, cls, inertia = k_means(X,K)
    
# Plot results:
figure(figsize=(14,9))
clusterplot(X, cls, centroids, y)
show()

print('Ran Exercise 10.1.1')