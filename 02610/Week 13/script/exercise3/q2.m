clear; close all; clc;
foo = @(x1, x2) 1/2*((x1-1).^2+x2.^2);

st = @(x2, beta) beta*x2.^2;

x =  -5:0.05:5;
y =  -5:0.05:5;
[X,Y] = meshgrid(x,y);
F = foo(X, Y);

figure;

contour(X,Y,F,100);
colorbar;
xlabel('x_1');
ylabel('x_2');
hold on;
plot(st(y, 0), y, 'rx');
hold on;
plot(st(y, 0.25), y, 'gx');
hold on;
plot(st(y, 1), y, 'bx');
hold on;
scatter(1, 0, 'k', 'filled');
hold off;

xlim([-5, 5]);

legend('contour', 'beta=0', 'beta=0.25', 'beta=1', 'global minimizer');
title('feasible set and function contour');