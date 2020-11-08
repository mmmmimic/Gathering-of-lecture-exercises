clear all, close all,

load data_exe3


a0 = [-1; -2];
[xopt,stat] = variable_projection(@fun_All,a0,t,y);

k = 0:stat.iter;
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);

figure,
subplot(1,2,1), semilogy(norm_df), title('|f''(x_k)|'),
subplot(1,2,2), semilogy(stat.F), title('f(x_k)'), 


figure, plot(t,y,'r.',t,xopt(1)*exp(xopt(3)*t)+xopt(2)*exp(xopt(4)*t),'b')