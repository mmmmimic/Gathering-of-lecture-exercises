# exercise 5.1.7
# requires Tree model from exercise 5.1.5
from ex5_1_6 import *

# Define a new data object (new type of wine) with the attributes given in the text
x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12]).reshape(1,-1)

# Evaluate the classification tree for the new data object
x_class = dtc.predict(x)[0]

# Print results
print('\nNew object attributes:')
for i in range(len(attributeNames)):
    print('{0}: {1}'.format(attributeNames[i],x[0][i]))
print('\nClassification result:')
print(classNames[x_class])

print('Ran Exercise 5.1.7')