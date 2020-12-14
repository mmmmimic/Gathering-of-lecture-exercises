% Question 3
x = linspace(-5, 5);
y = linspace(-5, 5);
[X, Y] = meshgrid(x, y);
f = X.^4 - 2*Y.*(X.^2) + Y.^2 + X.^2 - 2*X +5;
figure;
contour(X, Y, f, 1000);
colorbar;
xlabel('x_1');
ylabel('x_2');
title('contour plot of f(x)');