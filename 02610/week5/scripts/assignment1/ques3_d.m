clear;
fundfun = 'func2';
rho = 0.5;
c = 0.5;
maxit = 2000;
tol = 1e-8;
x0 = [0;2]; 
[x,stat] = newton_line(fundfun,x0, maxit, tol, rho, c);
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

%The Newton's method converges now. 

%It moved at very small steps in the first 80 iterations and then jumped to 
%a point close to the minimizer. Because in the first 80 iterations, Hession
%matrix was not definite, the step length had to be set in a smalll value to
%search slowly in a wrong direction until a definite Hession matrix appears.
%In initeration 81, Hession matrix was definite. The method was able to use 
%the Hession matrix that meets the assumptions.
% iter 80
[~,~,d2f] = func2([0.781372106996790;1.45204365026648])
% iter 81
[~,~,d2f] = func2([0.835675181387740;1.44752321376381])

%\alpha_k remained abound 0.01, the minimum \alpha value we set, in the 
%first 82 iterations. Then it increased a lot. 
%After 97 iterations, it raised to 1. 

%The function values f(x_k) is not monotonically decreasing. At iteration 
%82, there was a small increase on the function value. The reason is the 
%Newton's method doesn't promise to have a monotonic decrease in function 
%values, even though it's on the way to the minimizer. 
%The method only decides the correct direction to the optimal point. 

