import numpy as np

def categoric2numeric(x):
    '''
    CATEGORIC2NUMERIC converts data matrix with categorical columns given by
    numeric or text values to numeric columns using one out of K coding.

    Usage:
        X_num, attribute_names = categoric2numeric(x)

    Input:
        x                   categorical column of a data matrix 

    Output:
        X_num               Data matrix where categoric column has been
                            converted to one out of K coding
        attribute_names     list of string type with attribute names '''

    x = np.asarray(x).ravel()
    x_labels = np.unique(x)
    x_labels_str = x_labels.astype(str).tolist()
    N = len(x)
    M = len(x_labels)
    xc = np.zeros((N,M), dtype=int)
    for i in range(M):
        flags = x==x_labels[i]
        xc[flags,i] = 1
    return xc, x_labels_str
    
    
