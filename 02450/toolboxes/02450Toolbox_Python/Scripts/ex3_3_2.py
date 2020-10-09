# exercise 3.2.2

import numpy as np
from similarity import similarity

# Generate two data objects with M random attributes
M = 5;
x = np.random.rand(1,M)
y = np.random.rand(1,M)

# Two constants
a = 1.5
b = 1.5

# Check the statements in the exercise
print("Cosine scaling: %.4f " % (similarity(x,y,'cos') - similarity(a*x,y,'cos'))[0,0])
print("ExtendedJaccard scaling: %.4f " % (similarity(x,y,'ext') - similarity(a*x,y,'ext'))[0,0])
print("Correlation scaling: %.4f " % (similarity(x,y,'cor') - similarity(a*x,y,'cor'))[0,0])
print("Cosine translation: %.4f " % (similarity(x,y,'cos') - similarity(b+x,y,'cos'))[0,0])
print("ExtendedJaccard translation: %.4f " % (similarity(x,y,'ext') - similarity(b+x,y,'ext'))[0,0])
print("Correlation translation: %.4f " % (similarity(x,y,'cor') - similarity(b+x,y,'cor'))[0,0])

print('Ran Exercise 3.2.2')