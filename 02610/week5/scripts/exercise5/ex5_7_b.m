x0 = [1.2;1.2];
figure,
[x,stat] = steepestdescent_line('rosenbrock', x0);

subplot(2,1,1),
semilogy(0:stat.iter, stat.dX);
title('e_k of fixed_length SD')

subplot(2,1,2),
semilogy(0:stat.iter, stat.dX);
title('f(x_k) of fixed_length SD')

figure,
semilogy(stat.alpha, '.');
title('Alpha');

% In most cases, the values are approching 0.
% It doesn't always coverage, such as the case when length = 0.05 and 0.1.

x0 = [-1.2;1];
figure,
[x,stat] = steepestdescent_line('rosenbrock', x0);
subplot(2,1,1),
semilogy(0:stat.iter, stat.dX);
title('e_k of fixed_length SD')

subplot(2,1,2),
semilogy(0:stat.iter, stat.dX);
title('f(x_k) of fixed_length SD')
legend('1e-3', '1e-2', '5e-2', '1e-1');

figure,
semilogy(stat.alpha, '.');
title('Alpha');

% alpha changes in stages, like going downstairs. The solution is
% definitely closer to x* than anyone with fixed step length. 



