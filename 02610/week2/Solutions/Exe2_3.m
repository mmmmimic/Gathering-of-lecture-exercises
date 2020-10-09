x0 = 0.1;    % 0.5, 5, 
mu = 1.0;
alpha = 0.1;   % 1, 0.01
[xopt, stat] = steepestdescent(alpha, @PenFun1, x0, mu);

k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),


%% Newton

x0 = 0.1;    
mu = 1.0;
alpha = 1;   
[xopt, stat] = newton(alpha, @PenFun1, x0, mu);

k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(stat.dF), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),

