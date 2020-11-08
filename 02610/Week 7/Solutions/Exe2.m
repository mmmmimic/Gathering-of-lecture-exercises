clear all, close all,

lambda = 0;
x0 = 0.1;
flag_line = 1;
[xopt,stat] = GaussNewton_line(@fun_rJ_Q2,x0,flag_line,lambda);

k = 0:stat.iter;
err = abs(stat.X-0);
norm_df = abs(stat.dF);

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), semilogy(norm_df), title('|f''(x_k)|'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),


%%

lambda = 0.1;
x0 = 0.1;
flag_line = 1;
[xopt,stat] = GaussNewton_line(@fun_rJ_Q2,x0,flag_line,lambda);

k = 0:stat.iter;
err = abs(stat.X-0);
norm_df = abs(stat.dF);

figure,
subplot(2,2,1), semilogy(err), title('|x_k-x^*|'),
subplot(2,2,2), plot(err(2:end)./err(1:end-1)), title('err_{k+1}/err_{k}'),
subplot(2,2,3), semilogy(norm_df), title('|f''(x_k)|'),
subplot(2,2,4), semilogy(stat.F), title('f(x_k)'),


%%

lambda = -2;
x0 = 0.1;
flag_line = 0;
[xopt,stat] = GaussNewton_line(@fun_rJ_Q2,x0,flag_line,lambda);

k = 0:stat.iter;
err = abs(stat.X-0);
norm_df = abs(stat.dF);

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), semilogy(norm_df), title('|f''(x_k)|'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),

%%

lambda = -2;
x0 = 0.1;
flag_line = 1;
[xopt,stat] = GaussNewton_line(@fun_rJ_Q2,x0,flag_line,lambda);

k = 0:stat.iter;
err = abs(stat.X-0);
norm_df = abs(stat.dF);

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), semilogy(norm_df), title('|f''(x_k)|'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),

figure, plot(stat.alpha)