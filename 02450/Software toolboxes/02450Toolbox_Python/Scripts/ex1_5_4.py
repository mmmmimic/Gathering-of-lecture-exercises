## exercise 1.5.4
# Start by running the exercise 1.5.3 to load the Iris data in
# "classification format":
from ex1_5_3 import *

import numpy as np
import matplotlib.pyplot as plt

## Classification problem
# The current variables X and y represent a classification problem, in
# which a machine learning model will use the sepal and petal dimesions
# (stored in the matrix X) to predict the class (species of Iris, stored in
# the variable y). A relevant figure for this classification problem could
# for instance be one that shows how the classes are distributed based on
# two attributes in matrix X:
X_c = X.copy();
y_c = y.copy();
attributeNames_c = attributeNames.copy();
i = 1; j = 2;
color = ['r','g', 'b']
plt.title('Iris classification problem')
for c in range(len(classNames)):
    idx = y_c == c
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()
# Consider, for instance, if it would be possible to make a single line in
# the plot to delineate any two groups? Can you draw a line between
# the Setosas and the Versicolors? The Versicolors and the Virginicas?

## Regression problem
# Since the variable we wish to predict is petal length,
# petal length cannot any longer be in the data matrix X.
# The first thing we do is store all the information we have in the
# other format in one data matrix:
data = np.concatenate((X_c, np.expand_dims(y_c,axis=1)), axis=1)
# We need to do expand_dims to y_c for the dimensions of X_c and y_c to fit.

# We know that the petal length corresponds to the third column in the data
# matrix (see attributeNames), and therefore our new y variable is:
y_r = data[:, 2]

# Similarly, our new X matrix is all the other information but without the 
# petal length (since it's now the y variable):
X_r = data[:, [0, 1, 3, 4]]

# Since the iris class information (which is now the last column in X_r) is a
# categorical variable, we will do a one-out-of-K encoding of the variable:
species = np.array(X_r[:, -1], dtype=int).T
K = species.max()+1
species_encoding = np.zeros((species.size, K))
species_encoding[np.arange(species.size), species] = 1
# The encoded information is now a 150x3 matrix. This corresponds to 150
# observations, and 3 possible species. For each observation, the matrix
# has a row, and each row has two 0s and a single 1. The placement of the 1
# specifies which of the three Iris species the observations was.

# We need to replace the last column in X (which was the not encoded
# version of the species data) with the encoded version:
X_r = np.concatenate( (X_r[:, :-1], species_encoding), axis=1) 

# Now, X is of size 150x6 corresponding to the three measurements of the
# Iris that are not the petal length as well as the three variables that
# specifies whether or not a given observations is or isn't a certain type.
# We need to update the attribute names and store the petal length name 
# as the name of the target variable for a regression:
targetName_r = attributeNames_c[2]
attributeNames_r = np.concatenate((attributeNames_c[[0, 1, 3]], classNames), 
                                  axis=0)

# Lastly, we update M, since we now have more attributes:
N,M = X_r.shape

# A relevant figure for this regression problem could
# for instance be one that shows how the target, that is the petal length,
# changes with one of the predictors in X:
i = 2  
plt.title('Iris regression problem')
plt.plot(X_r[:, i], y_r, 'o')
plt.xlabel(attributeNames_r[i]);
plt.ylabel(targetName_r);
plt.show()
# Consider if you see a relationship between the predictor variable on the
# x-axis (the variable from X) and the target variable on the y-axis (the
# variable y). Could you draw a straight line through the data points for
# any of the attributes (choose different i)? 
# Note that, when i is 3, 4, or 5, the x-axis is based on a binary 
# variable, in which case a scatter plot is not as such the best option for 
# visulizing the information. 

