# exercise 5.1.4
# requires Tree model from exercise 5.1.2
from ex5_1_2 import *

# Define a new data object (a dragon) with the attributes given in the text
x = np.array([0, 2, 1, 2, 1, 1, 1]).reshape(1,-1)

# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]

# Print results
print('\nNew object attributes:')
print(dict(zip(attributeNames,x[0])))
print('\nClassification result:')
print(classNames[x_class])

print('Ran Exercise 5.1.4')