# Exercise 4.2.3

from matplotlib.pyplot import boxplot, xticks, ylabel, title, show

# requires data from exercise 4.2.1
import sys
sys.path.append('Software toolboxes/02450ToolBox_Python/Scripts/')
from ex4_2_1 import *

boxplot(X)
xticks(range(1,5),attributeNames)
ylabel('cm')
title('Fisher\'s Iris data set - boxplot')
show()

print('Ran Exercise 4.2.3')