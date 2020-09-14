import numpy as np

class BinClassifierEnsemble:
    '''
        Simple class to aggregate multiple weak classfiers into ensemble
    '''
    classifiers = []
    alpha = 0
    cn = 0

    def __init__(self, classifier_list, alpha='auto'):
        self.classifiers = classifier_list
        self.cn = len(self.classifiers)
        if type(alpha) is str and alpha=='auto':
            self.alpha = np.ones((self.cn,1),dtype=float)/self.cn
        else:
            self.alpha = np.asarray(alpha).ravel()
                
            
    def predict(self, X):
        '''
            Returns predicted class (value of y) for given X,
            based on ensemble majority vote.
        '''
        votes = np.zeros((X.shape[0],1))
        for c_id, c in enumerate(self.classifiers):
            y_est = np.mat(c.predict(X)).T
            y_est[y_est>1]=1 # restrict to binomial (or first-vs-rest)
            votes = votes + y_est*self.alpha[c_id]
        return (votes.astype(float)>.5).astype(int)
        
    def predict_proba(self, X):
        '''
            Returns proportion of ensemble votes for class being y=1,
            for given X, that is: votes1/(votes0+votes1).
        '''
        votes = np.ones((X.shape[0],1))
        for c_id, c in enumerate(self.classifiers):
            y_est = np.mat(c.predict(X)).T
            y_est[y_est>1]=1 # restrict to binomial (or first-vs-rest)
            votes = votes - y_est*self.alpha[c_id]
        return votes.astype(float)
