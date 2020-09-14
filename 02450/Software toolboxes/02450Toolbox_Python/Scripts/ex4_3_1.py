# exercise 4.3.1

from matplotlib.pyplot import (figure, title, boxplot, xticks, subplot, hist,
                               xlabel, ylim, yticks, show)
import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/wine.mat')
X = mat_data['X']
y = mat_data['y'].squeeze()
C = mat_data['C'][0,0]
M = mat_data['M'][0,0]
N = mat_data['N'][0,0]
attributeNames = [name[0][0] for name in mat_data['attributeNames']]
classNames = [cls[0][0] for cls in mat_data['classNames']]

# We start with a box plot of each attribute
figure()
title('Wine: Boxplot')
boxplot(X)
xticks(range(1,M+1), attributeNames, rotation=45)

# From this it is clear that there are some outliers in the Alcohol
# attribute (10x10^14 is clearly not a proper value for alcohol content)
# However, it is impossible to see the distribution of the data, because
# the axis is dominated by these extreme outliers. To avoid this, we plot a
# box plot of standardized data (using the zscore function).
figure(figsize=(12,6))
title('Wine: Boxplot (standarized)')
boxplot(zscore(X, ddof=1), attributeNames)
xticks(range(1,M+1), attributeNames, rotation=45)

# This plot reveals that there are clearly some outliers in the Volatile
# acidity, Density, and Alcohol attributes, i.e. attribute number 2, 8,
# and 11. 

# Next, we plot histograms of all attributes.
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram')
    

# This confirms our belief about outliers in attributes 2, 8, and 11.
# To take a closer look at this, we next plot histograms of the 
# attributes we suspect contains outliers
figure(figsize=(14,9))
m = [1, 7, 10]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X[:,m[i]],50)
    xlabel(attributeNames[m[i]])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('Wine: Histogram (selected attributes)')


# The histograms show that there are a few very extreme values in these
# three attributes. To identify these values as outliers, we must use our
# knowledge about the data set and the attributes. Say we expect volatide
# acidity to be around 0-2 g/dm^3, density to be close to 1 g/cm^3, and
# alcohol percentage to be somewhere between 5-20 % vol. Then we can safely
# identify the following outliers, which are a factor of 10 greater than
# the largest we expect.
outlier_mask = (X[:,1]>20) | (X[:,7]>10) | (X[:,10]>200)
valid_mask = np.logical_not(outlier_mask)

# Finally we will remove these from the data set
X = X[valid_mask,:]
y = y[valid_mask]
N = len(y)


# Now, we can repeat the process to see if there are any more outliers
# present in the data. We take a look at a histogram of all attributes:
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('Wine: Histogram (after outlier detection)')

# This reveals no further outliers, and we conclude that all outliers have
# been detected and removed.

show()

print('Ran Exercise 4.3.1')