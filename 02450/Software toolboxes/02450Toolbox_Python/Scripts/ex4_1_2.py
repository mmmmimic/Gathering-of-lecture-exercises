# exercise 4.1.2

from matplotlib.pyplot import (figure, title, subplot, plot, hist, show)
import numpy as np
# Number of samples
N = 200

# Mean
mu = 17

# Standard deviation
s = 2

# Number of bins in histogram
nbins = 20

# Generate samples from the Normal distribution
X = np.random.normal(mu,s,N).T 
# or equally:
X = np.random.randn(N).T * s + mu

# Plot the samples and histogram
figure()
title('Normal distribution')
subplot(1,2,1)
plot(X,'x')
subplot(1,2,2)
hist(X, bins=nbins)

# Compute empirical mean and standard deviation
mu_ = X.mean()
s_ = X.std(ddof=1)

print("Theoretical mean: ", mu)
print("Theoretical std.dev.: ", s)
print("Empirical mean: ", mu_)
print("Empirical std.dev.: ", s_)

show()

print('Ran Exercise 4.1.2')