x0 = [10;1];    
alpha = 0.05;   
[xopt, stat] = steepestdescent(alpha, @MvFun, x0);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),


%% Newton


x0 = [10;1];    
alpha = 1;   
[xopt, stat] = newton(alpha, @MvFun, x0);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),
