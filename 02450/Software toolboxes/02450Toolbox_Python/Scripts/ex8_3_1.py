# exercise 8.3.1 Fit neural network classifiers using softmax output weighting
from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import torch
# Load Matlab data file and extract variables of interest
mat_data = loadmat('../Data/synth1.mat')
X = mat_data['X']
X = X - np.ones((X.shape[0],1)) * np.mean(X,0)
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]

N, M = X.shape
C = len(classNames)
#%% Model fitting and prediction

# Define the model structure
n_hidden_units = 5 # number of hidden units in the signle hidden layer
model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            # Output layer:
                            # H hidden units to C classes
                            # the nodes and their activation before the transfer 
                            # function is often referred to as logits/logit output
                            torch.nn.Linear(n_hidden_units, C), # C logits
                            # To obtain normalised "probabilities" of each class
                            # we use the softmax-funtion along the "class" dimension
                            # (i.e. not the dimension describing observations)
                            torch.nn.Softmax(dim=1) # final tranfer function, normalisation of logit output
                            )
# Since we're training a multiclass problem, we cannot use binary cross entropy,
# but instead use the general cross entropy loss:
loss_fn = torch.nn.CrossEntropyLoss()
# Train the network:
net, _, _ = train_neural_net(model, loss_fn,
                             X=torch.tensor(X_train, dtype=torch.float),
                             y=torch.tensor(y_train, dtype=torch.long),
                             n_replicates=3,
                             max_iter=10000)
# Determine probability of each class using trained network
softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
# Get the estimated class as the class with highest probability (argmax on softmax_logits)
y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy() 
# Determine errors
e = (y_test_est != y_test)
print('Number of miss-classifications for ANN:\n\t {0} out of {1}'.format(sum(e),len(e)))

predict = lambda x:  (torch.max(net(torch.tensor(x, dtype=torch.float)), dim=1)[1]).data.numpy() 
figure(1,figsize=(9,9))
visualize_decision_boundary(predict, [X_train, X_test], [y_train, y_test], attributeNames, classNames)
title('ANN decision boundaries')

show()

print('Ran Exercise 8.3.1')