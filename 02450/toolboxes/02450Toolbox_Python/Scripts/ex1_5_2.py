# exercise 1.5.2
import numpy as np

# You can read data from excel spreadsheets after installing and importing xlrd
# module. In most cases, you will need only few functions to accomplish it:
# open_workbook(), col_values(), row_values()
import xlrd
# If you need more advanced reference, or if you are interested how to write 
# data to excel files, see the following tutorial:
# http://www.simplistix.co.uk/presentations/python-excel.pdf}

# Load xls sheet with data
# There's only a single sheet in the .xls, so we take out that sheet
doc = xlrd.open_workbook('../Data/iris.xls').sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(rowx=0, start_colx=0, end_colx=4)
# Try calling help(doc.row_values). You'll see that the above means
# that we extract columns 0 through 4 from the first row of the document, 
# which contains the header of the xls files (where the attributen names are)

# Extract class names to python list, then encode with integers (dict) just as 
# we did previously. The class labels are in the 5th column, in the rows 2 to 
# and up to 151:
classLabels = doc.col_values(4,1,151) # check out help(doc.col_values)
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames,range(len(classNames))))

# Extract vector y, convert to NumPy array
y = np.array([classDict[value] for value in classLabels])

# Preallocate memory, then extract data to matrix X
X = np.empty((150,4))
for i in range(4):
    X[:,i] = np.array(doc.col_values(i,1,151)).T

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
