## exercise 0.4.5
import numpy as np

# Setup two ararys
x = np.arange(1,6)
y = np.arange(2,12,2)
# Have a look at them by typing 'x' and 'y' in the console

# There's a difference between matrix multiplication and elementwise 
# multiplication, and specifically in Python its also important if you 
# are using the multiply operator "*" on an array object or a matrix object!

# Use the * operator to multiply the two arrays:
x*y

# Now, convert the arrays into matrices - 
x = np.asmatrix(np.arange(1,6))
y = np.asmatrix(np.arange(2,12,2))
# Again, have a look at them by typing 'x' and 'y' in the console

# Try using the * operator just as before now:
x*y 
# You should now get an error - try to explain why.

# array and matrix are two data structures added by NumPy package to the list of
# basic data structures in Python (lists, tuples, sets). We shall use both 
# array and matrix structures extensively throughout this course, therefore 
# make sure that you understand differences between them 
# (multiplication, dimensionality) and that you are able to convert them one 
# to another (asmatrix(), asarray() functions). 
# Generally speaking, array objects are used to represent scientific, numerical, 
# N-dimensional data. matrix objects can be very handy when it comes to 
# algebraic operations on 2-dimensional matrices.

# The ambiguity can be circumvented by using explicit function calls:
np.transpose(y)             # transposition/transpose of y
y.transpose()               # also transpose
y.T                         # also transpose

np.multiply(x,y)            # element-wise multiplication

np.dot(x,y.T)               # matrix multiplication
x @ y.T                     # also matrix multiplication


# There are various ways to make certain type of matrices.
a1 = np.array([[1, 2, 3], [4, 5, 6]])   # define explicitly
a2 = np.arange(1,7).reshape(2,3)        # reshape range of numbers
a3 = np.zeros([3,3])                    # zeros array
a4 = np.eye(3)                          # diagonal array
a5 = np.random.rand(2,3)                # random array
a6 = a1.copy()                          # copy
a7 = a1                                 # alias
m1 = np.matrix('1 2 3; 4 5 6; 7 8 9')   # define matrix by string
m2 = np.asmatrix(a1.copy())             # copy array into matrix
m3 = np.mat(np.array([1, 2, 3]))        # map array onto matrix
a8 = np.asarray(m1)                     # map matrix onto array
  
# It is easy to extract and/or modify selected items from arrays/matrices. 
# Here is how you can index matrix elements:
m = np.matrix('1 2 3; 4 5 6; 7 8 9')
m[0,0]		# first element
m[-1,-1]	# last element
m[0,:]		# first row
m[:,1]		# second column
m[1:3,-1]	# view on selected rows&columns

# Similarly, you can selectively assign values to matrix elements or columns:
m[-1,-1] = 10000
m[0:2,-1] = np.matrix('100; 1000')
m[:,0] = 0

# Logical indexing can be used to change or take only elements that 
# fulfil a certain constraint, e.g.
m2[m2>0.5]          # display values in m2 that are larger than 0.5
m2[m2<0.5] = 0      # set all elements that are less than 0.5 to 0 

#Below, several examples of common matrix operations, 
# most of which we will use in the following weeks.
# First, define two matrices:
m1 = 10 * np.mat(np.ones([3,3]))
m2 = np.mat(np.random.rand(3,3))

m1+m2               # matrix summation
m1*m2               # matrix product
np.multiply(m1,m2)  # element-wise multiplication
m1>m2               # element-wise comparison
m3 = np.hstack((m1,m2))    # combine/concatenate matrices horizontally 
# note that this is not equivalent to e.g. 
#   l = [m1, m2]
# in which case l is a list, and l[0] is m1
m4 = np.vstack((m1,m2))    # combine/concatenate matrices vertically 
m3.shape            # shape of matrix
m3.mean()           # mean value of all the elements
m3.mean(axis=0)     # mean values of the columns
m3.mean(axis=1)     # mean values of the rows
m3.transpose()      # transpose, also: m3.T
m2.I                # compute inverse matrix
