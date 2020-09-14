# exercise 5.1.3
import numpy as np
from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
import matplotlib.pyplot as plt
from matplotlib.image import imread

# requires data from exercise 5.1.1
from ex5_1_1 import *

# Fit regression tree classifier, Gini split criterion, no pruning
criterion='entropy'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/N)
dtc = dtc.fit(X,y)

fname='tree_' + criterion
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)

# Depending on the platform, we handle the file differently, first for Linux 
# Mac
if system() == 'Linux' or system() == 'Darwin':
    import graphviz
    # Make a graphviz object from the file
    src=graphviz.Source.from_file(fname + '.gvz')
    print('\n\n\n To view the tree, write "src" in the command prompt \n\n\n')
    
# ... and then for Windows:
if system() == 'Windows':
    # N.B.: you have to update the path_to_graphviz to reflect the position you 
    # unzipped the software in!
    windows_graphviz_call(fname=fname,
                          cur_dir=getcwd(),
                          path_to_graphviz=r'C:\Program Files (x86)\Graphviz2.38')
    plt.figure(figsize=(12,12))
    plt.imshow(imread(fname + '.png'))
    plt.box('off'); plt.axis('off')
    plt.show()

print('Ran Exercise 5.1.3')