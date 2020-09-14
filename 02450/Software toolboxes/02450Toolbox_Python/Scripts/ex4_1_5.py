# exercise 4.1.5

from matplotlib.pyplot import (figure, title, subplot, plot, hist, show, 
                               xlabel, ylabel, xticks, yticks, colorbar, cm, 
                               imshow, suptitle)
import numpy as np

# Number of samples
N = 1000

# Standard deviation of x1
s1 = 2

# Standard deviation of x2
s2 = 3

# Correlation between x1 and x2
corr = 0.5

# Covariance matrix
S = np.matrix([[s1*s1, corr*s1*s2], [corr*s1*s2, s2*s2]])

# Mean
mu = np.array([13, 17])

# Number of bins in histogram
nbins = 20

# Generate samples from multivariate normal distribution
X = np.random.multivariate_normal(mu, S, N)


# Plot scatter plot of data
figure(figsize=(12,8))
suptitle('2-D Normal distribution')

subplot(1,2,1)
plot(X[:,0], X[:,1], 'x')
xlabel('x1'); ylabel('x2')
title('Scatter plot of data')

subplot(1,2,2)
x = np.histogram2d(X[:,0], X[:,1], nbins)
imshow(x[0], cmap=cm.gray_r, interpolation='None', origin='lower')
colorbar()
xlabel('x1'); ylabel('x2'); xticks([]); yticks([]);
title('2D histogram')

show()

print('Ran Exercise 4.1.5')