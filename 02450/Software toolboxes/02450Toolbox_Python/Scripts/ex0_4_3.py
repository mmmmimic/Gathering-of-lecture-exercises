## exercise 0.4.3

# In Python you need to 'import' packages and external functions before you can 
# use them. We can import NumPy (which enables us to work with matrices, among 
# other things) by writing 'import numpy as np'. 
# We load the package into the ``namespace'' np to reference it easily,
# now we can write 'np.sum(X)' instead of 'numpy.sum(X)'.
import numpy as np 

# Remember you can mark a part of the code and press
# F9 to run that part alone.

# define variable a with numbers in the range from 0 to 7 (not inclusive)
a = np.arange(start=0,stop=7) 

# define variable b with numbers in the range from 2 to 17 in steps of 4
b = np.arange(start=2,stop=17,step=4) 

# similar to b but without explicit decleration of the input arguments names
c = np.arange(100, 95, -1) 

d = np.arange(1.2, 1.9, 0.1) 

e = np.pi*np.arange(0,2.5,.5) 
