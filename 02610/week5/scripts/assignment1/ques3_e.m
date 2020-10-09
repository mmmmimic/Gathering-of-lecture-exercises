clear;
fundfun = 'func2';
rho = 0.5;
c = 0.5;
maxit = 2000;
tol = 1e-8;
x0 = [0;2]; 
[x,stat] = steepestdescent_line(fundfun,x0, maxit, tol, rho, c);
figure,
plot(0:stat.iter, stat.F, 'linewidth', 2);
xlabel('iterations');
ylabel('f(x_k)');

figure,
semilogy(0:stat.iter, stat.dF, 'linewidth', 2);
xlabel('iterations');
ylabel('||∇f(x_k)||∞');

figure,
plot(1:stat.iter, stat.alpha, 'linewidth', 2);
xlabel('iterations');
ylabel('alpha');

% It converged. 
% It converged faster than the Newton's method. 
