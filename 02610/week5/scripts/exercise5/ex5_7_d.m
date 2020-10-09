x0 = [1.2;1.2];
sl = [1e-3, 1e-2, 5e-2, 1e-1];% step length
figure,
[x,stat] = BFGSmethod_line('rosenbrock',x0);
subplot(2,1,1),
semilogy(0:stat.iter, stat.dX);
title('e_k of fixed_length SD')
subplot(2,1,2),
semilogy(0:stat.iter, stat.dX);
title('f(x_k) of fixed_length SD')
stat.iter

x1=-1:0.05:2;
x2=-1:0.05:2;
[X,Y]=meshgrid(x1,x2);
F=100*(Y-X.^2).^2+(1-X).^2;

figure,
v = [0:2:10, 10:10:100, 100:100:2500];
[c,h]=contour(X,Y,F,v,'linewidth',2);
hold on;
plot(stat.X(1, :), stat.X(2, :), 'r-x', 'linewidth', 2);
hold on;
scatter([1,1.2], [1,1.2], 'gx', 'linewidth', 2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),

% In most cases, the values are approching 0.
% It doesn't always coverage, such as the case when length = 0.05 and 0.1.

x0 = [-1.2;1];
sl = [1e-3, 1e-2, 5e-2, 1e-1];% step length
figure,
[x,stat] = BFGSmethod_line('rosenbrock',x0);
subplot(2,1,1),
semilogy(0:stat.iter, stat.dX);
title('e_k of fixed_length SD')
subplot(2,1,2),
semilogy(0:stat.iter, stat.dX);
title('f(x_k) of fixed_length SD')
stat.iter

figure,
v = [0:2:10, 10:10:100, 100:100:2500];
[c,h]=contour(X,Y,F,v,'linewidth',2);
hold on;
plot(stat.X(1, :), stat.X(2, :), 'r-x', 'linewidth', 2);
hold on;
scatter([-1.2,1], [1,1], 'gx', 'linewidth', 2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),