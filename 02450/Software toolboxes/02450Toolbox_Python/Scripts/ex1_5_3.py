# exercise 1.5.3
import numpy as np

from scipy.io import loadmat
# You can load the matlab data (matlab's m-file) to Python environment with 
# 'loadmat()' function imported from 'scipy.io' module. 
# The matlab workspace is loaded as a dictionary, with keys corresponding to 
# matlab variable names, and values to arrays representing matlab matrices.

# Load Matlab data file to python dict structure
iris_mat = loadmat('../Data/iris.mat', squeeze_me=True)
# The argument squeeze_me ensures that there the variables we get from the 
# MATLAB filed are not stored within "unneeded" array dimensions.

# You can check which variables are in the loaded dict by calling
# the function keys() for the dict:
#mat_data.keys()
# this will tell you that X, y, M, N and C are stored in the dictionary,
# as well as some extra information about e.g. the used MATLAB version.

# We'll extract the needed variables by using these keys:
X = iris_mat['X']
y = iris_mat['y']
M = iris_mat['M']
N = iris_mat['N']
C = iris_mat['C']
attributeNames = iris_mat['attributeNames']
classNames = iris_mat['classNames']

# Loading the Iris data from the .mat-file was quite easy, because all the work
# of putting it into the correct format was already done. This is of course 
# likely not the case for your own data, where you'll need to do something 
# similar to the two previous exercises. We will, however, sometimes in the 
# course use .mat-files in the exercises.