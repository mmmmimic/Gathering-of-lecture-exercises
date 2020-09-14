# exercise 11.3.2

import numpy as np
from matplotlib.pyplot import figure, bar, title, plot, show
from toolbox_02450 import gausKernelDensity

# Draw samples from mixture of gaussians (as in exercise 11.1.1)
N = 1000; M = 1
x = np.linspace(-10, 10, 50)
X = np.empty((N,M))
m = np.array([1, 3, 6]); s = np.array([1, .5, 2])
c_sizes = np.random.multinomial(N, [1./3, 1./3, 1./3])
for c_id, c_size in enumerate(c_sizes):
    X[c_sizes.cumsum()[c_id]-c_sizes[c_id]:c_sizes.cumsum()[c_id],:] = np.random.normal(m[c_id], np.sqrt(s[c_id]), (c_size,M))



# Estimate the optimal kernel density width, by leave-one-out cross-validation
widths = 2.0**np.arange(-10,10)
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
    f, log_f = gausKernelDensity(X, w)
    logP[i] = log_f.sum()
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# Estimate density for each observation not including the observation
# itself in the density estimate
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i]

# Display the index of the lowest density data object
print('Lowest density: {0} for data object: {1}'.format(density[0],i[0]))

# Plot density estimate of outlier score
figure(1)
bar(range(20),density[:20].reshape(-1,))
title('Density estimate')
figure(2)
plot(logP)
title('Optimal width')
show()

print('Ran Exercise 11.3.2')