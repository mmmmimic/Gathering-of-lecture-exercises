# exercise 5.2.5
from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

# requires data from exercise 5.1.4
from ex5_1_5 import *


# Split dataset into features and target vector
alcohol_idx = attributeNames.index('Alcohol')
y = X[:,alcohol_idx]

X_cols = list(range(0,alcohol_idx)) + list(range(alcohol_idx+1,len(attributeNames)))
X = X[:,X_cols]

# Additional nonlinear attributes
fa_idx = attributeNames.index('Fixed acidity')
va_idx = attributeNames.index('Volatile acidity')
Xfa2 = np.power(X[:,fa_idx],2).reshape(-1,1)
Xva2 = np.power(X[:,va_idx],2).reshape(-1,1)
Xfava = (X[:,fa_idx]*X[:,va_idx]).reshape(-1,1)
X = np.asarray(np.bmat('X, Xfa2, Xva2, Xfava'))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display plots
figure(figsize=(12,8))

subplot(2,1,1)
plot(y, y_est, '.g')
xlabel('Alcohol content (true)'); ylabel('Alcohol content (estimated)')

subplot(4,1,3)
hist(residual,40)

subplot(4,3,10)
plot(Xfa2, residual, '.r')
xlabel('Fixed Acidity ^2'); ylabel('Residual')

subplot(4,3,11)
plot(Xva2, residual, '.r')
xlabel('Volatile Acidity ^2'); ylabel('Residual')

subplot(4,3,12)
plot(Xfava, residual, '.r')
xlabel('Fixed*Volatile Acidity'); ylabel('Residual')

show()

print('Ran Exercise 5.2.5')