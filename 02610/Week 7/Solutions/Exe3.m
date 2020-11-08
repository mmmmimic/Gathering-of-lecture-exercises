clear all, close all,

load data_exe3

x0 = [1; -1; 1; 2];
flag_line = 1;
[xopt,stat] = GaussNewton_line(@fun_rJ_Q3,x0,flag_line,t,y);

k = 0:stat.iter;
%err = abs(stat.X-0);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2+stat.dF(3,:).^2+stat.dF(4,:).^2);

figure,
%subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,2,1), semilogy(norm_df), title('|f''(x_k)|'),
subplot(1,2,2), semilogy(stat.F), title('f(x_k)'), 

%%
x0 = [3; -3; 3; 3];
flag_line = 1;
[xopt,stat] = GaussNewton_line(@fun_rJ_Q3,x0,flag_line,t,y);

disp('The method failed, since J is not full rank, i.e., J''J becomes singular!')

%%

x0 = [3; -3; 3; 3];
[xopt,stat] = Levenberg_Marquardt(@fun_rJ_Q3,x0,t,y);

k = 0:stat.iter;
%err = abs(stat.X-0);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2+stat.dF(3,:).^2+stat.dF(4,:).^2);

figure,
%subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,2,1), semilogy(norm_df), title('|f''(x_k)|'),
subplot(1,2,2), semilogy(stat.F), title('f(x_k)'), 

err = sqrt((stat.X(1,:)-xopt(1)).^2+(stat.X(2,:)-xopt(2)).^2+(stat.X(3,:)-xopt(3)).^2+(stat.X(4,:)-xopt(4)).^2);
figure, semilogy(err(2:end)./err(1:end-1))

figure, plot(t,y,'r.',t,xopt(1)*exp(-xopt(3)*t)+xopt(2)*exp(-xopt(4)*t),'b')