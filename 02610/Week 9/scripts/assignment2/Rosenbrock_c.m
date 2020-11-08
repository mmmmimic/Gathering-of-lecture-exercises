%% Rosenbrock's Problem
clear; close; clc;

%% c) LM Method
fun_rJ = @fun_rJ_Rosen;
x0 = [-1.2; 1];
x_star = [1; 1];
tau = 1e-3;
[x, stat] = Levenberg_Marquardt_yq(fun_rJ, x0, tau);

tmp = stat.F;
for i = 1:length(stat.F)
    tmp(i) = norm(stat.X(:, i) - x_star, 2);
end

figure;
subplot(1,3,1)
plot(1:length(tmp), tmp, 'r-', 'linewidth', 1);
xlabel('iteration');
ylabel('e_k');

subplot(1,3,2)
plot(1:length(stat.F), stat.F, 'b-', 'linewidth', 1);
xlabel('iteration');
ylabel('f(x_k)');

tmp = stat.F;
for i = 1:length(stat.F)
    tmp(i) = norm(stat.dF(:, i), 2);
end
subplot(1,3,3)
plot(1:length(tmp), tmp, 'g-', 'linewidth', 1);
xlabel('iteration');
ylabel('||df(x_k)||_2');

r = [];
for i = 1:length(tmp)-1
   r = [r, tmp(i+1) / tmp(i)]; 
end
figure;
plot(1:18, r, 'k-', 'linewidth', 1);
xlabel('iteration');
ylabel('e(x_{k+1})/e(x_k)');
% 19 iterations
% Linear convergence rate -> Quadratic convergence rate