# Exercise 4.2.7

from matplotlib.pyplot import (figure, imshow, xticks, xlabel, ylabel, title, 
                               colorbar, cm, show)
from scipy.stats import zscore

# requires data from exercise 4.2.1
from ex4_2_1 import *

X_standarized = zscore(X, ddof=1)

figure(figsize=(12,6))
imshow(X_standarized, interpolation='none', aspect=(4./N), cmap=cm.gray);
xticks(range(4), attributeNames)
xlabel('Attributes')
ylabel('Data objects')
title('Fisher\'s Iris data matrix')
colorbar()

show()

print('Ran Exercise 4.2.7')