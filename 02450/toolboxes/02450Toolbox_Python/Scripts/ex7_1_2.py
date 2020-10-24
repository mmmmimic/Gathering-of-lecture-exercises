from toolbox_02450 import jeffrey_interval
from ex7_1_1 import *

# Compute the Jeffreys interval
alpha = 0.05
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

# The jeffrey interval relates with model accuracy

alpha = 0.1
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

alpha = 0.01
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha)

print("Theta point estimate", thetahatA, " CI: ", CIA)

# alpha relates with the range of interval. The smaller alpha is, the wider intervals are. 