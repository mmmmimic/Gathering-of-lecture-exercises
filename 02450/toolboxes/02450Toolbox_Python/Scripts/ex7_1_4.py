from toolbox_02450 import mcnemar
from ex7_1_1 import *

# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
# After changing model B to decision tree, model A is better than B. 

# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)

print("theta = theta_A-theta_C point estimate", thetahat, " CI: ", CI, "p-value", p)

# Model A is better than model C, but not better than model B (before changing). 