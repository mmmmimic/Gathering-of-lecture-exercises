close all, clear all

x = [10;10];  
[f, df, d2f] = Myfun2(x);

eta=0.2;
[xopt, stat] = TR_dogleg(eta,@Myfun2, x);

k = 0:stat.iter;
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,3,1), semilogy(err), title('||x_k-x^*||'),
subplot(1,3,2), semilogy(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),
