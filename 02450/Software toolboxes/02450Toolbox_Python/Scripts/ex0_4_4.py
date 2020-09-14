## exercise 0.4.4
import numpy as np
# Extracting the elements from vectors is easy. Consider the
# following definition of x and the echoed results
x = np.concatenate([np.zeros(2), np.arange(0, 3.6, 0.6), np.ones(3)])
x[1:5] # take out elements 2 through 5 (notice 6 is not included)
np.size(x) # return the size of x (equivalent to len(x) since x is an array)
len(x) # return the length of x


# Try writing help(len) and help(np.size)

x[-1] # take the last element of x
x[1::2] # return every other element of x starting from the 2nd

# The length of x is 11; what is x[11] - and why?

# Inserting numbers into vectors is also easy. Using the same
# definition of x and observe the results when typing
y = x;
y[1::2]  = np.pi 
# Notice that we're inserting the same scalar value "pi" into all elements 
# that we index y with

# You can also try:
#y[1::2] = np.arange(2,12,2)

# Observe the results when indexing the vector y with
# y[1] and y[0]. Is y[0] defined? 