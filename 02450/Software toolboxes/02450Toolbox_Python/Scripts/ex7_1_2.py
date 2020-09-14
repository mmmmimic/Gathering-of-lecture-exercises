from toolbox_02450 import jeffrey_interval
from ex7_1_1 import *

# Compute the Jeffreys interval
alpha = 0.05
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)
