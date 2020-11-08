%% Newton's Method
%% a) If the Jacobian J is not a square matrix, then can we still apply Newton' s method?
% If the Jacobian J is not a square matrix, we will not be able to apply
% Newton' s method. Because we can't calculate the inverse of Jacobian. 

%% b) Revise your implementation of Newton’s method for solving a minimizatoin problem 

%% c) run the function
alpha = 1;
fundfun = @func_rJ_exe;
x0 = [3;1];
[x,stat] = newton(alpha, fundfun, x0);

% Which convergence rate can you see? If the method did not converge 
% quadratically, what can be the reason?
figure;
plot(stat.X);
xlabel('iteration');
ylabel('e_k');

figure;
plot(stat.F);
xlabel('iteration');
ylabel('1/2||x_k||_2^2');

%% d) further observe the convergence rate
figure;
plot(stat.X1);
xlabel('iteration');
ylabel('x_1(k)');


X2_r = [];
x2 = stat.X2(1);
for i = 2:length(stat.X2)
   X2_r = [X2_r, stat.X2(i)/x2];
   x2 = stat.X2(i);
end

figure;
plot(X2_r);
xlabel('iteration');
ylabel('x_2(k+1)/x_2(k)');

%%
tmp = [];
for i = 2:length(stat.X)
   tmp = [tmp, stat.X(i) / stat.X(i-1)]; 
end
figure;
plot(tmp);
xlabel('iteration');
ylabel('e(x_{k+1})/e(x_k)');

% the ratio converged to 0.5, it's linear convergence. 