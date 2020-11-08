%% Levenberg-Marquardt method
clear; close; clc;

%% a) Call your implementation of the L-M method from Week 7
fun_rJ = @func_rJ_exe;
x0 = [3; 1];

[x,stat] = Levenberg_Marquardt(fun_rJ, x0);

figure;
plot(stat.X);
xlabel('iteration');
ylabel('e_k');

figure;
plot(stat.F);
xlabel('iteration');
ylabel('1/2||x_k||_2^2');

tmp = [];
for i = 2:length(stat.X)
   tmp = [tmp, stat.X(i) / stat.X(i-1)]; 
end
figure;
plot(tmp);
xlabel('iteration');
ylabel('e(x_{k+1})/e(x_k)');

% It's linear convergence
%% b) implement Levenberg Marquardt yq.m
fun_rJ = @func_rJ_exe;
x0 = [3; 1];

[x,stat] = Levenberg_Marquardt_yq(fun_rJ, x0, 1);

figure;
plot(stat.X);
xlabel('iteration');
ylabel('e_k');

figure;
plot(stat.F);
xlabel('iteration');
ylabel('1/2||x_k||_2^2');

tmp = [];
for i = 2:length(stat.X)
   tmp = [tmp, stat.X(i) / stat.X(i-1)]; 
end
figure;
plot(tmp);
xlabel('iteration');
ylabel('e(x_{k+1})/e(x_k)');

% The later one is better, although they are all linear convergence
%% c) What can be the reason for stalling between iteration 20-30?
% I found that between around iteration 20-30, p(step length) are very
% small, nearly 0. It's the instant reason. For x, x1 is pretty small while
% x2 is still decreasing. Plus, lambda is decreasing in 
% this period, meaning that Gauss-Newton method is turning to Steepest Gradient
% method, but not yet. That's the reason why the iteration process seems to stall. 


