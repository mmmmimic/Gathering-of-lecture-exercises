clear;
fundfun = 'func2';
alpha = 1;
maxit = 10;
tol = 1e-30; % strict criteria for convergence
x0 = [0;2]; 
[x,stat] = newton(alpha,fundfun, x0, maxit, tol);
x1 =  -3:0.05:3;
x2 =  -3:0.05:3;
[X1,X2]=meshgrid(x1,x2);
F=X1.^4/4-X1.^2+2*X1+(X2-1).^2;

figure,
v = -5:34;
[c,h]=contour(X1,X2,F,v,'linewidth',2);
hold on;
plot(stat.X(1,:) ,stat.X(2,:), 'r-x', 'linewidth', 2);
hold on;
scatter([stat.X(1,1), stat.X(1,end)] ,[stat.X(2,1), stat.X(2,end)], 'gx', 'linewidth', 2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),

stat.X

[~, ~, d2f] = func2(x0);
d2f
[~, ~, d2f] = func2([0;1]);
d2f
% The method didn't converge. 
% When in point [0,2] and [0,1], the Hession matrix is not definite. 
% It doesn't meet the assumption that the Newton's method is based on. 
% That's why the method didn't converge. 