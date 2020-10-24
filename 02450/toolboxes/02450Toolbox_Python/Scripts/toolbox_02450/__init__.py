'''
 Collection of functions and tools for the needs of 02450 Introduction to 
 Machine Learning course.
'''
__version__ = 'Revision: 2019-02-11'

import sklearn.metrics.cluster as cluster_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from matplotlib.pyplot import contourf
from matplotlib import cm
from toolbox_02450.statistics import *

def remove_zero_cols(m):
    '''Function removes from given matrix m the column vectors containing only zeros.'''
    rows = range(m.shape[0])
    cols = np.nonzero(sum(abs(m)))[1].tolist()[0]
    return m[np.ix_(rows,cols)]

def remove_zero_rows(m):
    '''Function removes from given matrix m the row vectors containing only zeros.'''
    rows = np.nonzero(sum(abs(m.T)).T)[0].tolist()[0]
    cols = range(m.shape[1])
    return m[np.ix_(rows,cols)]

def remove_zero_rows_and_cols(m):
    '''Function removes from given matrix m the row vectors and the column vectors containing only zeros.'''
    rows = np.nonzero(sum(abs(m.T)).T)[0].tolist()[0]
    cols = np.nonzero(sum(abs(m)))[1].tolist()[0]
    return m[np.ix_(rows,cols)]


def bmplot(yt, xt, X):
    ''' Function plots matrix X as image with lines separating fields. '''
    plt.imshow(X,interpolation='none',cmap='bone')
    plt.xticks(range(0,len(xt)), xt)
    plt.yticks(range(0,len(yt)), yt)
    for i in range(0,len(yt)):
        plt.axhline(i-0.5, color='black')
    for i in range(0,len(xt)):
        plt.axvline(i-0.5, color='black')


def glm_validate(X,y,cvf=10):
    ''' Validate linear regression model using 'cvf'-fold cross validation.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns MSE averaged over 'cvf' folds.

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds        
    '''
    y = y.squeeze()
    CV = model_selection.KFold(n_splits=cvf, shuffle=True)
    validation_error=np.empty(cvf)
    f=0
    for train_index, test_index in CV.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        m = linear_model.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        validation_error[f] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        f=f+1
    return validation_error.mean()
        

def feature_selector_lr(X,y,cvf=10,features_record=None,loss_record=None,display=''):
    ''' Function performs feature selection for linear regression model using
        'cvf'-fold cross validation. The process starts with empty set of
        features, and in every recurrent step one feature is added to the set
        (the feature that minimized loss function in cross-validation.)

        Parameters:
        X       training data set
        y       vector of values
        cvf     number of crossvalidation folds

        Returns:
        selected_features   indices of optimal set of features
        features_record     boolean matrix where columns correspond to features
                            selected in subsequent steps
        loss_record         vector with cv errors in subsequent steps
        
        Example:
        selected_features, features_record, loss_record = ...
            feature_selector_lr(X_train, y_train, cvf=10)
            
    ''' 
    y = y.squeeze() #ÆNDRING JLH #9/3
    # first iteration error corresponds to no-feature estimator
    if loss_record is None:
        loss_record = np.array([np.square(y-y.mean()).sum()/y.shape[0]])
    if features_record is None:
        features_record = np.zeros((X.shape[1],1))

    # Add one feature at a time to find the most significant one.
    # Include only features not added before.
    selected_features = features_record[:,-1].nonzero()[0]
    min_loss = loss_record[-1]
    if display is 'verbose':
        print(min_loss)
    best_feature = False
    for feature in range(0,X.shape[1]):
        if np.where(selected_features==feature)[0].size==0:
            trial_selected = np.concatenate((selected_features,np.array([feature])),0).astype(int)
            # validate selected features with linear regression and cross-validation:
            trial_loss = glm_validate(X[:,trial_selected],y,cvf)
            if display is 'verbose':
                print(trial_loss)
            if trial_loss<min_loss:
                min_loss = trial_loss 
                best_feature = feature

    # If adding extra feature decreased the loss function, update records
    # and go to the next recursive step
    if best_feature is not False:
        features_record = np.concatenate((features_record, np.array([features_record[:,-1]]).T), 1)
        features_record[best_feature,-1]=1
        loss_record = np.concatenate((loss_record,np.array([min_loss])),0)
        selected_features, features_record, loss_record = feature_selector_lr(X,y,cvf,features_record,loss_record)
        
    # Return current records and terminate procedure
    return selected_features, features_record, loss_record
        

def rlr_validate(X,y,lambdas,cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds     
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    CV = model_selection.KFold(cvf, shuffle=True)
    M = X.shape[1]
    w = np.empty((M,cvf,len(lambdas)))
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
        
        # precompute terms
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0,len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0,0] = 0 # remove bias regularization
            w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
            # Evaluate training and test performance
            train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
            test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
        
def dbplotf(X,y,fun,grid_range,resolution=100.0) :     
    # smoothness of color-coding:
    levels = 100
    # convert from one-out-of-k encoding, if neccessary:
    if np.ndim(y)>1: y = np.argmax(y,1)
    # compute grid range if not given explicitly:
    if grid_range=='auto':
        grid_range = [X.min(axis=0)[0], X.max(axis=0)[0], X.min(axis=0)[1], X.max(axis=0)[1]]
        
    delta_f1 = np.float(grid_range[1]-grid_range[0])/float(resolution)
    delta_f2 = np.float(grid_range[3]-grid_range[2])/float(resolution)
    f1 = np.arange(grid_range[0],grid_range[1],delta_f1)
    f2 = np.arange(grid_range[2],grid_range[3],delta_f2)
    F1, F2 = np.meshgrid(f1, f2)
    C = len(np.unique(y).tolist())
    # adjust color coding:
    if C==2: C_colors = ['b', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)']; C_levels = [.5]
    if C==3: C_colors = ['b', 'g', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)', 'Class C (y=2)']; C_levels = [.66, 1.34]
    if C==4: C_colors = ['b', 'w', 'y', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)', 'Class C (y=2)', 'Class D (y=3)']; C_levels = [.74, 1.5, 2.26]
    if C>4:
        # One way to get class colors for more than 4 classes. Note this may result in illegible figures!
        C_colors=[]
        C_legend=[]
        for c in range(C):
            C_colors.append(plt.cm.jet.__call__(c*255/(C-1))[:3])
            C_legend.append('Class {0}'.format(c))
        C_levels = [.74, 1.5, 2.26]
    
    coords = np.mat( [[f1[i], f2[j]] for i in range(len(f1)) for j in range(len(f2))] )
    values_list = fun(coords)#np.mat(classifier.predict(coords))
    if np.ndim(values_list)>1: raise ValueError('Expected vector got something else')
    if len(set(values_list))==1: raise ValueError('Expect multiple predicted value, but all predictions are equal. Try a more complex model')
    
    
    if values_list.shape[0]!=len(f1)*len(f2): values_list = values_list.T
    
                        
    values = np.asarray(np.reshape(values_list,(len(f1),len(f2))).T)
            
    #hold(True)
    for c in range(C):
        cmask = (y==c); plt.plot(X[cmask,0], X[cmask,1], '.', color=C_colors[c], markersize=10)
    plt.title('Model prediction and decision boundary')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2');
    plt.contour(F1, F2, values, levels=C_levels, colors=['k'], linestyles='dashed')
    plt.contourf(F1, F2, values, levels=np.linspace(values.min(),values.max(),levels), cmap=plt.cm.jet, origin='image')
    plt.colorbar(format='%.1f'); plt.legend(C_legend)
    #hold(False) 
 
def dbplot(classifier, X, y, grid_range, resolution=100):
    ''' Plot decision boundry for given binomial or multinomial classifier '''

    # smoothness of color-coding:
    levels = 100
    # convert from one-out-of-k encoding, if neccessary:
    if np.ndim(y)>1: y = np.argmax(y,1)
    # compute grid range if not given explicitly:
    if grid_range=='auto':
        grid_range = [X.min(0)[0], X.max(0)[0], X.min(0)[1], X.max(0)[1]]
        
    delta_f1 = np.float(grid_range[1]-grid_range[0])/resolution
    delta_f2 = np.float(grid_range[3]-grid_range[2])/resolution
    f1 = np.arange(grid_range[0],grid_range[1],delta_f1)
    f2 = np.arange(grid_range[2],grid_range[3],delta_f2)
    F1, F2 = np.meshgrid(f1, f2)
    C = len(np.unique(y).tolist())
    # adjust color coding:
    if C==2: C_colors = ['b', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)']; C_levels = [.5]
    if C==3: C_colors = ['b', 'g', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)', 'Class C (y=2)']; C_levels = [.66, 1.34]
    if C==4: C_colors = ['b', 'w', 'y', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)', 'Class C (y=2)', 'Class D (y=3)']; C_levels = [.74, 1.5, 2.26]
    if C>4:
        # One way to get class colors for more than 4 classes. Note this may result in illegible figures!
        C_colors=[]
        C_legend=[]
        for c in range(C):
            C_colors.append(plt.cm.jet.__call__(c*255/(C-1))[:3])
            C_legend.append('Class {0}'.format(c))
        C_levels = [.74, 1.5, 2.26]

    coords = np.array( [[f1[i], f2[j]] for i in range(len(f1)) for j in range(len(f2))] )
    values_list = classifier.predict(coords)
    if values_list.shape[0]!=len(f1)*len(f2): values_list = values_list.T
    values = np.reshape(values_list,(len(f1),len(f2))).T
            
    #hold(True)
    for c in range(C):
        cmask = (y==c); plt.plot(X[cmask,0], X[cmask,1], '.', color=C_colors[c], markersize=10)
    plt.title('Model prediction and decision boundary')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2');
    plt.contour(F1, F2, values, levels=C_levels, colors=['k'], linestyles='dashed')
    plt.contourf(F1, F2, values, levels=np.linspace(values.min(),values.max(),levels), cmap=plt.cm.jet, origin='image')
    plt.colorbar(format='%.1f'); plt.legend(C_legend)
    #hold(False)


def dbprobplot(classifier, X, y, grid_range, resolution=100):
    ''' Plot decision boundry for given binomial classifier '''

    # smoothness of color-coding:
    levels = 100
    # convert from one-out-of-k encoding, if neccessary:
    if np.ndim(y)>1: y = np.argmax(y,1)
    # compute grid range if not given explicitly:
    if grid_range=='auto':
        grid_range = [X.min(0)[0], X.max(0)[0], X.min(0)[1], X.max(0)[1]]
    # if more than two classes, display the first class against the rest:
    y[y>1]=1        
    C=2; C_colors = ['b', 'r']; C_legend = ['Class A (y=0)', 'Class B (y=1)']; C_levels = [.5]
        
    delta_f1 = np.float(grid_range[1]-grid_range[0])/resolution
    delta_f2 = np.float(grid_range[3]-grid_range[2])/resolution
    f1 = np.arange(grid_range[0],grid_range[1],delta_f1)
    f2 = np.arange(grid_range[2],grid_range[3],delta_f2)
    F1, F2 = np.meshgrid(f1, f2)

    coords = np.array([[f1[i], f2[j]] for i in range(len(f1)) for j in range(len(f2))])
    values_list = classifier.predict_proba(coords)
    if values_list.shape[0]!=len(f1)*len(f2): values_list = values_list.T
    values_list = 1-values_list[:,0] # probability of class being y=1
    values = np.reshape(values_list,(len(f1),len(f2))).T
           
    #hold(True)
    for c in range(C):
        cmask = (y==c); plt.plot(X[cmask,0], X[cmask,1], '.', color=C_colors[c], markersize=10)
    plt.title('Model prediction and decision boundary')
    plt.xlabel('Feature 1'); plt.ylabel('Feature 2');
    
    plt.contour(F1, F2, values, levels=C_levels, colors=['k'], linestyles='dashed')
    contourf(F1, F2, values, levels=np.linspace(values.min(),values.max(),levels), cmap=cm.jet, origin='image')
    plt.colorbar(format='%.1f'); plt.legend(C_legend)
    #hold(False)

from sklearn import metrics

def rocplot(p, y):
    '''
    function: AUC, TPR, FPR = rocplot(p, y)
    ROCPLOT Plots the receiver operating characteristic (ROC) curve and
    calculates the area under the curve (AUC). 

    Notice that the function assumes values of p are all distinct. 

    
    Usage:
        rocplot(p, y)
        AUC, TPR, FDR = rocplot(p, y)
 
     Input: 
         p: Estimated probability of class 1. (Between 0 and 1.)
         y: True class indices. (Equal to 0 or 1.)

    Output:
        AUC: The area under the ROC curve
        TPR: True positive rate
        FPR: False positive rate
    '''
    #ind = np.argsort(p,0)
    #x = y[ind].A.ravel()
    #FNR = np.mat(np.cumsum(x==1, 0, dtype=float)).T / np.sum(x==1,0)
    #TPR = 1 - FNR
    #TNR = np.mat(np.cumsum(x==0, 0, dtype=float)).T / np.sum(x==0,0)
    #FPR = 1 - TNR
    #onemat = np.mat([1]) 
    #TPR = np.bmat('onemat; TPR'); FPR = np.mat('onemat; FPR') # Don't get this line.
    #TPR = vstack( (np.ones(1), TPR))
    #FPR = vstack( (np.ones(1), FPR))
    
    #AUC = -np.diff(FPR,axis=0).T * (TPR[0:-1]+TPR[1:])/2
    #AUC = AUC[0,0]    

    #%%
    fpr, tpr, thresholds = metrics.roc_curve(y,p)
    #FPR = fpr 
    #TPR = TPR
    #TPR
    AUC = metrics.roc_auc_score(y, p)
    #%%
    plt.plot(fpr, tpr, 'r', [0, 1], [0, 1], 'k')
    plt.grid()
    plt.xlim([-0.01,1.01]); plt.ylim([-0.01,1.01])
    plt.xticks(np.arange(0,1.1,.1)); plt.yticks(np.arange(0,1.1,.1))
    plt.xlabel('False positive rate (1-Specificity)')
    plt.ylabel('True positive rate (Sensitivity)')
    plt.title('Receiver operating characteristic (ROC)\n AUC={:.3f}'.format(AUC))    
    
    
    return AUC, tpr, fpr
    
    


def confmatplot(y_true, y_est):
    '''
    The function plots confusion matrix for classification results. 
    
    Usage:
        confmatplot(y_true, y_estimated)
 
     Input: 
         y_true: Vector of true class labels.
         y_estimated: Vector of estimated class labels.
    '''
    from sklearn.metrics import confusion_matrix
    y_true = np.asarray(y_true).ravel(); y_est = np.asarray(y_est).ravel()
    C = np.unique(y_true).shape[0]
    cm = confusion_matrix(y_true, y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    plt.imshow(cm, cmap='binary', interpolation='None');
    plt.colorbar(format='%.2f')
    plt.xticks(range(C)); plt.yticks(range(C));
    plt.xlabel('Predicted class'); plt.ylabel('Actual class');
    plt.title('Confusion matrix (Accuracy: {:}%, Error Rate: {:}%)'.format(accuracy, error_rate));
    

def bootstrap(X, y, N, weights='auto'):
    '''
    function: X_bs, y_bs = bootstrap(X, y, N, weights)
    The function extracts the bootstrap set from given matrices X and y.
    The distribution of samples is determined by weights parameter
    (default: 'auto', equal weights). 
    
    Usage:
        X_bs, y_bs = bootstrap(X, y, N, weights)
 
     Input: 
         X: Estimated probability of class 1. (Between 0 and 1.)
         y: True class indices. (Equal to 0 or 1.)
         N: number of samples to be drawn
         weights: probability of occurence of samples (default: equal)

    Output:
        X_bs: Matrix with rows drawn randomly from X wrt given distribution
        y_bs: Matrix with rows drawn randomly from y wrt given distribution
    '''
    if type(weights) is str and weights == 'auto':
        weights = np.ones((X.shape[0],1),dtype=float)/X.shape[0]
    else:
        weights = np.array(weights,dtype=float)
        weights = (weights/weights.sum()).ravel().tolist()

    #bc = np.random.multinomial(N, weights, 1).ravel()
    
    #selected_indices = [] 
    #while bc.sum()>0:
    #     selected_indices += np.where(bc>0)[0].tolist(); bc[bc>0]-=1
    #np.random.shuffle(selected_indices)
        
    selected_indices = np.random.choice(range(N), size=(N,1), replace=True,p=weights).flatten()
    if np.ndim(y)==1:
        return X[selected_indices, :], y[selected_indices]
    else:
        return X[selected_indices, :], y[selected_indices, :]
    
  

def clusterplot(X, clusterid, centroids='None', y='None', covars='None'):
    '''
    CLUSTERPLOT Plots a clustering of a data set as well as the true class
    labels. If data is more than 2-dimensional it should be first projected
    onto the first two principal components. Data objects are plotted as a dot
    with a circle around. The color of the dot indicates the true class,
    and the cicle indicates the cluster index. Optionally, the centroids are
    plotted as filled-star markers, and ellipsoids corresponding to covariance
    matrices (e.g. for gaussian mixture models).

    Usage:
    clusterplot(X, clusterid)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix)
    clusterplot(X, clusterid, centroids=c_matrix, y=y_matrix, covars=c_tensor)
    
    Input:
    X           N-by-M data matrix (N data objects with M attributes)
    clusterid   N-by-1 vector of cluster indices
    centroids   K-by-M matrix of cluster centroids (optional)
    y           N-by-1 vector of true class labels (optional)
    covars      M-by-M-by-K tensor of covariance matrices (optional)
    '''
    
    X = np.asarray(X)
    cls = np.asarray(clusterid)
    if type(y) is str and y=='None':
        y = np.zeros((X.shape[0],1))
    else:
        y = np.asarray(y)
    if type(centroids) is not str:
        centroids = np.asarray(centroids)
    K = np.size(np.unique(cls))
    C = np.size(np.unique(y))
    ncolors = np.max([C,K])
    
    # plot data points color-coded by class, cluster markers and centroids
    #hold(True)
    colors = [0]*ncolors
    for color in range(ncolors):
        colors[color] = plt.cm.jet(color/(ncolors-1))[:3]
    for i,cs in enumerate(np.unique(y)):
        plt.plot(X[(y==cs).ravel(),0], X[(y==cs).ravel(),1], 'o', markeredgecolor='k', markerfacecolor=colors[i],markersize=6, zorder=2)
    for i,cr in enumerate(np.unique(cls)):
        plt.plot(X[(cls==cr).ravel(),0], X[(cls==cr).ravel(),1], 'o', markersize=12, markeredgecolor=colors[i], markerfacecolor='None', markeredgewidth=3, zorder=1)
    if type(centroids) is not str:        
        for cd in range(centroids.shape[0]):
            plt.plot(centroids[cd,0], centroids[cd,1], '*', markersize=22, markeredgecolor='k', markerfacecolor=colors[cd], markeredgewidth=2, zorder=3)
    # plot cluster shapes:
    if type(covars) is not str:
        for cd in range(centroids.shape[0]):
            x1, x2 = gauss_2d(centroids[cd],covars[cd,:,:])
            plt.plot(x1,x2,'-', color=colors[cd], linewidth=3, zorder=5)
    #hold(False)

    # create legend        
    legend_items = np.unique(y).tolist()+np.unique(cls).tolist()+np.unique(cls).tolist()
    for i in range(len(legend_items)):
        if i<C: legend_items[i] = 'Class: {0}'.format(legend_items[i]);
        elif i<C+K: legend_items[i] = 'Cluster: {0}'.format(legend_items[i]);
        else: legend_items[i] = 'Centroid: {0}'.format(legend_items[i]);
    plt.legend(legend_items, numpoints=1, markerscale=.75, prop={'size': 9})


def gauss_2d(centroid, ccov, std=2, points=100):
    ''' Returns two vectors representing slice through gaussian, cut at given standard deviation. '''
    mean = np.c_[centroid]; tt = np.c_[np.linspace(0, 2*np.pi, points)]
    x = np.cos(tt); y=np.sin(tt); ap = np.concatenate((x,y), axis=1).T
    d, v = np.linalg.eig(ccov); d = std * np.sqrt(np.diag(d))
    bp = np.dot(v, np.dot(d, ap)) + np.tile(mean, (1, ap.shape[1])) 
    return bp[0,:], bp[1,:]
    
def clusterval(y, clusterid):
    '''
    CLUSTERVAL Estimate cluster validity using Entropy, Purity, Rand Statistic,
    and Jaccard coefficient.
    
    Usage:
      Entropy, Purity, Rand, Jaccard = clusterval(y, clusterid);
    
    Input:
       y         N-by-1 vector of class labels 
       clusterid N-by-1 vector of cluster indices
    
    Output:
      Entropy    Entropy measure.
      Purity     Purity measure.
      Rand       Rand index.
      Jaccard    Jaccard coefficient.
    '''
    NMI = cluster_metrics.supervised.normalized_mutual_info_score(y,clusterid)
    
    #y = np.asarray(y).ravel(); clusterid = np.asarray(clusterid).ravel()
    C = np.unique(y).size; K = np.unique(clusterid).size; N = y.shape[0]
    EPS = 2.22e-16
    
    p_ij = np.zeros((K,C))          # probability that member of i'th cluster belongs to j'th class
    m_i = np.zeros((K,1))           # total number of objects in i'th cluster
    for k in range(K):
        m_i[k] = (clusterid==k).sum()
        yk = y[clusterid==k]
        for c in range(C):
            m_ij = (yk==c).sum()    # number of objects of j'th class in i'th cluster
            p_ij[k,c] = m_ij.astype(float)/m_i[k]
    entropy = ( (1-(p_ij*np.log2(p_ij+EPS)).sum(axis=1))*m_i.T ).sum() / (N*K) 
    purity = ( p_ij.max(axis=1) ).sum() / K

    f00=0; f01=0; f10=0; f11=0
    for i in range(N):
        for j in range(i):
            if y[i]!=y[j] and clusterid[i]!=clusterid[j]: f00 += 1;     # different class, different cluster    
            elif y[i]==y[j] and clusterid[i]==clusterid[j]: f11 += 1;   # same class, same cluster
            elif y[i]==y[j] and clusterid[i]!=clusterid[j]: f10 += 1;   # same class, different cluster    
            else: f01 +=1;                                              # different class, same cluster
    rand = np.float(f00+f11)/(f00+f01+f10+f11)
    jaccard = np.float(f11)/(f01+f10+f11)

    return rand, jaccard, NMI

    
def gausKernelDensity(X,width):
    '''
    GAUSKERNELDENSITY Calculate efficiently leave-one-out Gaussian Kernel Density estimate
    Input: 
      X        N x M data matrix
      width    variance of the Gaussian kernel
    
    Output: 
      density        vector of estimated densities
      log_density    vector of estimated log_densities
    '''
    X = np.mat(np.asarray(X))
    N,M = X.shape

    # Calculate squared euclidean distance between data points
    # given by ||x_i-x_j||_F^2=||x_i||_F^2-2x_i^Tx_j+||x_i||_F^2 efficiently
    x2 = np.square(X).sum(axis=1)
    D = x2[:,[0]*N] - 2*X.dot(X.T) + x2[:,[0]*N].T

    # Evaluate densities to each observation
    Q = np.exp(-1/(2.0*width)*D)
    # do not take density generated from the data point itself into account
    Q[np.diag_indices_from(Q)]=0
    sQ = Q.sum(axis=1)
    
    density = 1/((N-1)*np.sqrt(2*np.pi*width)**M+1e-100)*sQ
    log_density = -np.log(N-1)-M/2*np.log(2*np.pi*width)+np.log(sQ)
    return np.asarray(density), np.asarray(log_density)



def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):
    """
    Train a neural network with PyTorch based on a training set consisting of
    observations X and class y. The model and loss_fn inputs define the
    architecture to train and the cost-function update the weights based on,
    respectively.
    
    Usage:
        Assuming loaded dataset (X,y) has been split into a training and 
        test set called (X_train, y_train) and (X_test, y_test), and
        that the dataset has been cast into PyTorch tensors using e.g.:
            X_train = torch.tensor(X_train, dtype=torch.float)
        Here illustrating a binary classification example based on e.g.
        M=2 features with H=2 hidden units:
    
        >>> # Define the overall architechture to use
        >>> model = lambda: torch.nn.Sequential( 
                    torch.nn.Linear(M, H),  # M features to H hiden units
                    torch.nn.Tanh(),        # 1st transfer function
                    torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()      # final tranfer function
                    ) 
        >>> loss_fn = torch.nn.BCELoss() # define loss to use
        >>> net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3)
        >>> y_test_est = net(X_test) # predictions of network on test set
        >>> # To optain "hard" class predictions, threshold the y_test_est
        >>> See exercise ex8_2_2.py for indepth example.
        
        For multi-class with C classes, we need to change this model to e.g.:
        >>> model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, H), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            torch.nn.Linear(H, C), # H hidden units to C classes
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        
        And the final class prediction is based on the argmax of the output
        nodes:
        >>> y_class = torch.max(y_test_est, dim=1)[1]
        
    Args:
        model:          A function handle to make a torch.nn.Sequential.
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary 
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        max_iter:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerenace:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
                        
        
    Returns:
        A list of three elements:
            best_net:       A trained torch.nn.Sequential that had the lowest 
                            loss of the trained replicates
            final_loss:     An float specifying the loss of best performing net
            learning_curve: A list containing the learning curve of the best net.
    
    """
    
    import torch
    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        print('\n\tReplicate: {}/{}'.format(r+1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights) 
        net = model()
        
        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to 
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        #optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)
        
        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):
            y_est = net(X) # forward pass, predict labels on training set
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            
        # display final loss
        print('\t\tFinal loss:')
        print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
        print(print_str)
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve

def get_data_ranges(x):
    '''
    Determine minimum and maximum for each feature in input x and output as 
    numpy array.
    
    Args:
            x:          An array of shape (N,M), where M corresponds to 
                        features and N corresponds to observations.
                        
    Returns:
            ranges:     A numpy array of minimum and maximum values for each  
                        feature dimension.
    '''
    N, M = x.shape
    ranges = []
    for m in range(M):
        ranges.append(np.min(x[:,m]))
        ranges.append(np.max(x[:,m]))
    return np.array(ranges)

def visualize_decision_boundary(predict, 
                                 X, y, 
                                 attribute_names,
                                 class_names,
                                 train=None, test=None, 
                                 delta=5e-3,
                                 show_legend=True):
    '''
    Visualize the decision boundary of a classifier trained on a 2 dimensional
    input feature space.
    
    Creates a grid of points based on ranges of features in X, then determines
    classifier output for each point. The predictions are color-coded and plotted
    along with the data and a visualization of the partitioning in training and
    test if provided.
    
    Args:
        predict:
                A lambda function that takes the a grid of shape [M, N] as 
                input and returns the prediction of the classifier. M corre-
                sponds to the number of features (M==2 required), and N corre-
                sponding to the number of points in the grid. Can e.g. be a 
                trained PyTorch network (torch.nn.Sequential()), such as trained
                using toolbox_02450.train_neural_network, where the provided
                function would be something similar to: 
                >>> predict = lambda x: (net(torch.tensor(x, dtype=torch.float))).data.numpy()
                
        X:      A numpy array of shape (N, M), where N is the number of 
                observations and M is the number of input features (constrained
                to M==2 for this visualization).
                If X is a list of len(X)==2, then each element in X is inter-
                preted as a partition of training or test data, such that 
                X[0] is the training set and X[1] is the test set.
                
        y:      A numpy array of shape (N, 1), where N is the number of 
                observations. Each element is either 0 or 1, as the 
                visualization is constrained to a binary classification
                problem.
                If y is a list of len(y)==2, then each element in y is inter-
                preted as a partion of training or test data, such that 
                y[0] is the training set and y[1] is the test set. 
                
        attribute_names:
                A list of strings of length 2 giving the name
                of each of the M attributes in X.
                
        class_names: 
                A list of strings giving the name of each class in y.
                
        train (optional):  
                A list of indices describing the indices in X and y used for
                training the network. E.g. from the output of:
                    sklearn.model_selection.KFold(2).split(X, y)
                    
        test (optional):   
                A list of indices describing the indices in X and y used for
                testing the network (see also argument "train").
                
        delta (optional):
                A float describing the resolution of the decision
                boundary (default: 0.01). Default results grid of 100x100 that
                covers the first and second dimension range plus an additional
                25 percent.
        show_legend (optional):
                A boolean designating whether to display a legend. Defaults
                to True.
                
    Returns:
        Plots the decision boundary on a matplotlib.pyplot figure.
        
    '''
    
    import torch
    
    C = len(class_names)
    if isinstance(X, list) or isinstance(y, list):
        assert isinstance(y, list), 'If X is provided as list, y must be, too.'
        assert isinstance(y, list), 'If y is provided as list, X must be, too.'
        assert len(X)==2, 'If X is provided as a list, the length must be 2.'
        assert len(y)==2, 'If y is provided as a list, the length must be 2.'
        
        N_train, M = X[0].shape
        N_test, M = X[1].shape
        N = N_train+N_test
        grid_range = get_data_ranges(np.concatenate(X))
    else:
        N, M = X.shape
        grid_range = get_data_ranges(X)
    assert M==2, 'TwoFeatureError: Current neural_net_decision_boundary is only implemented for 2 features.'
    # Convert test/train indices to boolean index if provided:
    if train is not None or test is not None:
        assert not isinstance(X, list), 'Cannot provide indices of test and train partition, if X is provided as list of train and test partition.'
        assert not isinstance(y, list), 'Cannot provide indices of test and train partition, if y is provided as list of train and test partition.'
        assert train is not None, 'If test is provided, then train must also be provided.'
        assert test is not None, 'If train is provided, then test must also be provided.'
        train_index = np.array([(int(e) in train) for e in np.linspace(0, N-1, N)])
        test_index = np.array([(int(e) in test) for e in np.linspace(0, N-1, N)])
    
    xx = np.arange(grid_range[0], grid_range[1], delta)
    yy = np.arange(grid_range[2], grid_range[3], delta)
    # make a mesh-grid from a and b that spans the grid-range defined
    grid = np.stack(np.meshgrid(xx, yy))
    # reshape grid to be of shape "[number of feature dimensions] by [number of points in grid]"
    # this ensures that the shape fits the way the network expects input to be shaped
    # and determine estimated class label for entire featurespace by estimating
    # the label of each point in the previosly defined grid using provided
    # function predict()
    grid_predictions = predict(np.reshape(grid, (2,-1)).T)
    
    # Plot data with color designating class and transparency+shape
    # identifying partition (test/train)
    if C == 2:
        c = ['r','b']
        cmap = cm.bwr
        vmax=1
    else:
        c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        cmap = cm.tab10
        vmax=10
        
    s = ['o','x']; t = [.33, 1.0];
    for i in range(C):
        if train is not None and test is not None:
            for j, e in enumerate([train_index, test_index]):
                idx = (np.squeeze(y)==i) & e
                plt.plot(X[idx, 0], X[idx, 1], s[j],color=c[i], alpha=t[j])
        if isinstance(X,list) and isinstance(y, list):
            for j, (X_par, y_par) in enumerate(zip(X,y)):
                idx = np.squeeze(y_par)==i
                h = plt.plot(X_par[idx, 0], X_par[idx, 1],s[j], color=c[i], alpha=t[j])
  
    plt.xlim(grid_range[0:2])
    plt.ylim(grid_range[2:])
    plt.xlabel(attribute_names[0]);
    plt.ylabel(attribute_names[1])

    # reshape the predictions for each point in the grid to be shaped like
    # an image that corresponds to the feature-scace using the ranges that
    # defined the grid (a and b)
    decision_boundary = np.reshape(grid_predictions, (len(yy), len(xx)))
    # display the decision boundary
    ax = plt.imshow(decision_boundary, cmap=cmap, 
           extent=grid_range, vmin=0, vmax=vmax, alpha=.33, origin='lower')
    plt.axis('auto')
    if C == 2:
        plt.contour(grid[0], grid[1], decision_boundary, levels=[.5])
        plt.colorbar(ax, fraction=0.046, pad=0.04);
    if show_legend:
        plt.legend([class_names[i]+' '+e for i in range(C) for e in ['train','test']],
                   bbox_to_anchor=(1.2,1.0))
def draw_neural_net(weights, biases, tf, 
                    attribute_names = None,
                    figsize=(12, 12),
                    fontsizes=(15, 12)):
    '''
    Draw a neural network diagram using matplotlib based on the network weights,
    biases, and used transfer-functions. 
    
    :usage:
        >>> w = [np.array([[10, -1], [-8, 3]]), np.array([[7], [-1]])]
        >>> b = [np.array([1.5, -8]), np.array([3])]
        >>> tf = ['linear','linear']
        >>> draw_neural_net(w, b, tf)
    
    :parameters:
        - weights: list of arrays
            List of arrays, each element in list is array of weights in the 
            layer, e.g. len(weights) == 2 with a single hidden layer and
            an output layer, and weights[0].shape == (2,3) if the input 
            layer is of size two (two input features), and there are 3 hidden
            units in the hidden layer.
        - biases: list of arrays
            Similar to weights, each array in the list defines the bias
            for the given layer, such that len(biases)==2 signifies a 
            single hidden layer, and biases[0].shape==(3,) signifies that
            there are three hidden units in that hidden layer, for which
            the array defines the biases of each hidden node.
        - tf: list of strings
            List of strings defining the utilized transfer-function for each 
            layer. For use with e.g. neurolab, determine these by:
                tf = [type(e).__name__ for e in transfer_functions],
            when the transfer_functions is the parameter supplied to 
            nl.net.newff, e.g.:
                [nl.trans.TanSig(), nl.trans.PureLin()]
        - (optional) figsize: tuple of int
            Tuple of two int designating the size of the figure, 
            default is (12, 12)
        - (optional) fontsizes: tuple of int
            Tuple of two ints giving the font sizes to use for node-names and
            for weight displays, default is (15, 12).
        
    Gist originally developed by @craffel and improved by @ljhuang2017
    [https://gist.github.com/craffel/2d727968c3aaebd10359]
    
    Modifications (Nov. 7, 2018):
        * adaption for use with 02450
        * display coefficient sign and magnitude as color and 
          linewidth, respectively
        * simplifications to how the method in the gist was called
        * added optinal input of figure and font sizes
        * the usage example how  implements a recreation of the Figure 1 in
          Exercise 8 of in the DTU Course 02450
    '''

   
   
    #Determine list of layer sizes, including input and output dimensionality
    # E.g. layer_sizes == [2,2,1] has 2 inputs, 2 hidden units in a single 
    # hidden layer, and 1 outout.
    layer_sizes = [e.shape[0] for e in weights] + [weights[-1].shape[1]]
    
    # Internal renaming to fit original example of figure.
    coefs_ = weights
    intercepts_ = biases

    # Setup canvas
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(); ax.axis('off');

    # the center of the leftmost node(s), rightmost node(s), bottommost node(s),
    # topmost node(s) will be placed here:
    left, right, bottom, top = [.1, .9, .1, .9]
    
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Determine normalization for width of edges between nodes:
    largest_coef = np.max([np.max(np.abs(e)) for e in weights])
    min_line_width = 1
    max_line_width = 5
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  
                  lw =1, head_width=0.01, head_length=0.02)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 
                                v_spacing/8.,
                                color='w', ec='k', zorder=4)
            if n == 0:
                if attribute_names:
                    node_str = str(attribute_names[m])
                    
                else:
                    node_str = r'$X_{'+str(m+1)+'}$'
                plt.text(left-0.125, layer_top - m*v_spacing+v_spacing*0.1, node_str,
                         fontsize=fontsizes[0])
            elif n == n_layers -1:
                node_str =  r'$y_{'+str(m+1)+'}$'
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing,
                         node_str, fontsize=fontsizes[0])
                if m==layer_size-1:
                    tf_str = 'Transfer-function: \n' + tf[n-1]
                    plt.text(n*h_spacing + left, bottom, tf_str, 
                             fontsize=fontsizes[0])
            else:
                node_str = r'$H_{'+str(m+1)+','+str(n)+'}$'
                plt.text(n*h_spacing + left+0.00, 
                         layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing),
                         node_str, fontsize=fontsizes[0])
                if m==layer_size-1:
                    tf_str = 'Transfer-function: \n' + tf[n-1]
                    plt.text(n*h_spacing + left, bottom, 
                             tf_str, fontsize=fontsizes[0])
            ax.add_artist(circle)
            
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., 
                                color='w', ec='k', zorder=4)
            plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01), 
                     y_bias, r'$1$', fontsize=fontsizes[0])
            ax.add_artist(circle)   
            
    # Edges
    # Edges between nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                colour = 'g' if coefs_[n][m, o]>0 else 'r'
                linewidth = (coefs_[n][m, o] / largest_coef)*max_line_width+min_line_width
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], 
                                  c=colour, linewidth=linewidth)
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                plt.text( xm1, ym1,\
                         str(round(coefs_[n][m, o],4)),\
                         rotation = rot_mo_deg, \
                         fontsize = fontsizes[1])
                
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            colour = 'g' if intercepts_[n][o]>0 else 'r'
            linewidth = (intercepts_[n][o] / largest_coef)*max_line_width+min_line_width
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], 
                          c=colour,
                          linewidth=linewidth)
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
            xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
            yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
            plt.text( xo1, yo1,\
                 str(round(intercepts_[n][o],4)),\
                 rotation = rot_bo_deg, \
                 fontsize = fontsizes[1])    
                
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
        
    plt.show()

def windows_graphviz_call(fname, cur_dir, path_to_graphviz):
    from subprocess import call

    call_str = path_to_graphviz + r'\bin\dot'+ \
                ' -Tpng '+ cur_dir +\
                '\\' + fname + '.gvz '+\
                '-o' + cur_dir + \
                '\\' + fname +'.png' +\
                ' -Gdpi=600'
    call(call_str)
    
