%% Meyer's problem
clear; close; clc;
%% b) Call the Matlab function Levenberg Marquardt
fun_rJ = @fun_rJ_Meyer2;
x0 = [8.85; 4; 2.5];
tau = 1;
y = [34780, 28610, 23650, 19630, 16370,...
    13720, 11540, 9744, 8261, 7030, 6005, 5147, 4427,...
    3820, 3307, 2872]';
u = 0.45+0.05.*[1:16]';
[x, stat] = Levenberg_Marquardt_yq(fun_rJ, x0, tau, u, y);

figure;
plot(1:89, stat.F, '-r', 'lineWidth', 1);
xlabel('iteration');
ylabel('f(x_k)');

norm_df = stat.F;
for i = 1:length(norm_df)
    norm_df(i) = norm(stat.dF(i), 2);
end
figure;
plot(1:89, norm_df, '-b', 'lineWidth', 1);
xlabel('iteration');
ylabel('||df(x_k)||_2');


% 88 iterations are needed
% This one needs much less iterations than the former one. 