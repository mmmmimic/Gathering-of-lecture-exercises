# exercise 5.1.5
import numpy as np
from scipy.io import loadmat

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].astype(int).squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]

attributeNames = [i[0][0] for i in mat_data['attributeNames']]
classNames = [j[0] for i in mat_data['classNames'] for j in i]


# Remove outliers
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)
X = X[valid_mask,:]
y = y[valid_mask]
# Remove attribute 12 (Quality score)
X = X[:,0:11]
attributeNames = attributeNames[0:11]
# Update N and M
N, M = X.shape

print('Ran Exercise 5.1.5')