import numpy as np
from scipy.stats import zscore


def similarity(X, Y, method):
    '''
    SIMILARITY Computes similarity matrices

    Usage:
        sim = similarity(X, Y, method)

    Input:
    X   N1 x M matrix
    Y   N2 x M matrix 
    method   string defining one of the following similarity measure
           'SMC', 'smc'             : Simple Matching Coefficient
           'Jaccard', 'jac'         : Jaccard coefficient 
           'ExtendedJaccard', 'ext' : The Extended Jaccard coefficient
           'Cosine', 'cos'          : Cosine Similarity
           'Correlation', 'cor'     : Correlation coefficient

    Output:
    sim Estimated similarity matrix between X and Y
        If input is not binary, SMC and Jaccard will make each
        attribute binary according to x>median(x)

    Copyright, Morten Morup and Mikkel N. Schmidt
    Technical University of Denmark '''

    X = np.mat(X)
    Y = np.mat(Y)
    N1, M = np.shape(X)
    N2, M = np.shape(Y)
    
    method = method[:3].lower()
    if method=='smc': # SMC
        X,Y = binarize(X,Y);
        sim = ((X*Y.T)+((1-X)*(1-Y).T))/M
    elif method=='jac': # Jaccard
        X,Y = binarize(X,Y);
        sim = (X*Y.T)/(M-(1-X)*(1-Y).T)        
    elif method=='ext': # Extended Jaccard
        XYt = X*Y.T
        sim = XYt / (np.log( np.exp(sum(np.power(X.T,2))).T * np.exp(sum(np.power(Y.T,2))) ) - XYt)
    elif method=='cos': # Cosine
        sim = (X*Y.T)/(np.sqrt(sum(np.power(X.T,2))).T * np.sqrt(sum(np.power(Y.T,2))))
    elif method=='cor': # Correlation
        X_ = zscore(X,axis=1,ddof=1)
        Y_ = zscore(Y,axis=1,ddof=1)
        sim = (X_*Y_.T)/(M-1)
    return sim
        
def binarize(X,Y=None):
    ''' Force binary representation of the matrix, according to X>median(X) '''
    x_was_transposed = False
    if Y is None:
        if X.shape[0] == 1:
            x_was_transposed = True
            X = X.T;
        
        Xmedians = np.ones((np.shape(X)[0],1)) * np.median(X,0)
        Xflags = X>Xmedians
        X[Xflags] = 1; X[~Xflags] = 0

        if x_was_transposed:
            return X.T
        return X
    else:
        #X = np.matrix(X); Y = np.matrix(Y);
        #XYmedian= np.median(np.bmat('X; Y'),0)
        #Xmedians = np.ones((np.shape(X)[0],1)) * XYmedian
        #Xflags = X>Xmedians
        #X[Xflags] = 1; X[~Xflags] = 0
        #Ymedians = np.ones((np.shape(Y)[0],1)) * XYmedian
        #Yflags = Y>Ymedians
        #Y[Yflags] = 1; Y[~Yflags] = 0
        return [binarize(X,None),binarize(Y,None)]
        

## Example
#import numpy as np
#from similarity import binarize2
#A = np.asarray([[1,2,3,4,5],[6,7,8,9,10],[1,2,3,4,5],[6,7,8,9,10]]).T
#binarize2(A,['a','b','c','d'])
def binarize2(X,columnnames):
    X = np.concatenate((binarize(X),1-binarize(X)),axis=1)

    new_column_names = []
    [new_column_names.append(elm) for elm in [name+' 50th-100th percentile' for name in columnnames]]
    [new_column_names.append(elm) for elm in [name+' 0th-50th percentile' for name in columnnames]]

    return X, new_column_names