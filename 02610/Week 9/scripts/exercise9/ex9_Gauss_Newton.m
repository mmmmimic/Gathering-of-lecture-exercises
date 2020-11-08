%% Gauss-Newton Method

%% a) Show that if J(x_k) is a square matrix and nonsingular, the Gauss-Newton
% iteration step is identical to the Newton step.

% x_(k+1) = x_k - (J(x_k)^T * J(x_k))^(-1) * J(x_k)^T * r(x_k)
% When J(x_k) is a square matrix and nonsigular, 
% then, J(x_k)^(-1) and (J(x_k)^T)^(-1) exist

% x_(k+1) = x_k - J(x_k)^(-1) * (J(x_k)^T)^(-1) * J(x_k)^T * r(x_k)
% x_(k+1) = x_k - J(x_k)^(-1) * r(x_k)
% It's identical to the Newton step

%% b) If the Jacobian J is not a square matrix, can we still apply the Gauss-Newton method?
% If Jacobian J is not a square matrix, let's assume its size to be (n x m)
% , then (J(x_k)^T * J(x_k)) shall have a size of (m x n x n x m) = (m x m).
% Thus, (J(x_k)^T * J(x_k)) is still a square matrix. As long as its
% determinant is not 0, it's inversable. Then we can still apply the Gauss-Newton method


%% c) call Gauss-Newton Line
flag_line = false;
x0 = [3; 1];
fun_rJ = @func_rJ_exe;
[x, stat] = GaussNewton_line(fun_rJ, x0, flag_line);

figure;
plot(stat.X);
xlabel('iteration');
ylabel('e_k');

figure;
plot(stat.F);
xlabel('iteration');
ylabel('1/2||x_k||_2^2');

% The result is exactly the same with that of Newton method. Because J(x_k)
% is always a square matrix and nonsingular here. It's the case we proved
% in the first question. 
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