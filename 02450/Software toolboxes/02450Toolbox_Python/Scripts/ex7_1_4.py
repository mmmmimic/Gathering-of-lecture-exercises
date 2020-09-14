from toolbox_02450 import mcnemar
from ex7_1_1 import *

# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
