clear all; close all; clc;
%% ex2.3.1 (a)
x0 = 0.1;
alpha = 0.1;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);

k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%% ex2.3.1 (b)
x0 = 5;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%%
x0 = 0.5;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%% ex2.3.1 (c)
x0 = 5;
alpha = 0.5;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%% ex2.3.2
x0 = 0.1;
alpha = 1;
[x,stat] = newton(alpha,@PenFun1,x0,1);
k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%% ex2.3.3
x0 = 0.01;
alpha = 0.01;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%%
x0 = 0.01;
alpha = 0.01;
[x,stat] = newton(alpha,@PenFun1,x0,1);
k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', err', abs(stat.dF)', stat.F'];

figure,
subplot(1,3,1), semilogy(err), title('|x_k-x^*|'),
subplot(1,3,2), plot(abs(stat.dF)), title('f''(x_k)'),
subplot(1,3,3), plot(stat.F), title('f(x_k)'),
%%
% steepest descent converages faster here. 
% but when it is approaching the minimizer, the speed becomes very slow
% newton method doesn't have this problem, but it is more computational
% expensive. And in the early stage, it's slow. 

%% ex2.4.1
[f,df,ddf] = MyFun([5,10]);
%% ex2.4.2
x0 = [10;1];
alpha = 0.05;
[x,stat] = steepestdescent(alpha,@MyFun,x0);
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);
figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),
%% ex2.4.3
x0 = [10;1];
alpha = 1;
[x,stat] = newton(alpha,@MyFun,x0);
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);
figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),
%%
% In this question, Newton method is appearantly much faster than Steepest
% descent method, even though it needs to compute the Hession matrix. 

%% ex2.5.1
[x,stat] = steepestdescent_line(1,@MyFun,x0,0.5,0.1);
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);
figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),
%% ex2.5.2
[x,stat] = newton_line(1,@MyFun,x0,0.5,0.1);
err = sqrt(stat.X(1,:).^2+stat.X(2,:).^2);
norm_df = sqrt(stat.dF(1,:).^2+stat.dF(2,:).^2);
figure,
subplot(1,3,1), plot(err), title('||x_k-x^*||'),
subplot(1,3,2), plot(norm_df), title('||f''(x_k)||'),
subplot(1,3,3), semilogy(stat.F), title('f(x_k)'),
% Now the result is much better
disp('Since it is a quadratic function, Newton''s method only need 2 iterations.')
disp('For this example, there are no difference for Newton''s method with or without line search.')
disp('But we need note that with line search Newton''s method can have global convergence.')