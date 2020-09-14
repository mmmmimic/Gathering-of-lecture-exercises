# exercise 4.2.1

import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook('../Data/iris.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0,0,4)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(4,1,151)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy matrix and transpose
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((150,4))
for i in range(4):
    X[:,i] = np.array(doc.col_values(i,1,151)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)

print('Ran Exercise 4.2.1')