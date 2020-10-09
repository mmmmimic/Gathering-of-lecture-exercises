clear, close, clc;
% initilization
x0 = [1.2;1.2];

% optimization
% SGD
[~, stat1] = steepestdescent_line('rosenbrock', x0);
% Newton
[~,stat2] = newton(1,'rosenbrock', x0);
% BFGS
[~,stat3] = BFGSmethod_line('rosenbrock',x0);

% e_k
figure,
loglog(0:stat1.iter, stat1.dX, 'r', 'linewidth', 2);
hold on;
loglog(0:stat2.iter, stat2.dX, 'g', 'linewidth', 2);
hold on;
loglog(0:stat3.iter, stat3.dX, 'b', 'linewidth', 2);
legend('SGD','Newton','BFGS');
xlabel('iteration number')
ylabel('e_k');


% f(x_k)
figure,
loglog(0:stat1.iter, stat1.F, 'r', 'linewidth', 2);
hold on;
loglog(0:stat2.iter, stat2.F, 'g', 'linewidth', 2);
hold on;
loglog(0:stat3.iter, stat3.F, 'b', 'linewidth', 2);
legend('SGD','Newton','BFGS');
xlabel('iteration number')
ylabel('f(x_k)');