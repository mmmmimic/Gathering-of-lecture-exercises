# install apyori using a standard method: conda install apyori or pip install apyori
from apyori import apriori
# Load resources from previous exercise
import numpy as np
from ex12_1_3 import X,labels
# ex12_1_4
# This is a helper function that transforms a binary matrix into transactions.
# Note the format used for courses.txt was (nearly) in a transaction format,
# however we will need the function later which is why we first transformed
# courses.txt to our standard binary-matrix format.
def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

# apyori requires data to be in a transactions format, forunately we just wrote a helper function to do that.
T = mat2transactions(X,labels)
rules = apriori( T, min_support=0.8, min_confidence=1)

# This function print the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y
def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules
# Print rules found in the courses file.
print_apriori_rules(rules)