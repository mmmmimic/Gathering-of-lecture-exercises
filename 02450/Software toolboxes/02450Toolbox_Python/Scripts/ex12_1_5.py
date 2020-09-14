# ex12_1_5
# Load data from the wine dataset
from scipy.io import loadmat
mat_data = loadmat('../Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
attributeNames = [name[0][0] for name in mat_data['attributeNames']]

# We will now transform the wine dataset into a binary format. Notice the changed attribute names:
from similarity import binarize2
Xbin, attributeNamesBin = binarize2(X, attributeNames)
print("X, i.e. the wine dataset, has now been transformed into:")
print(Xbin)
print(attributeNamesBin)