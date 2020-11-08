%% plot figures for the report
clear; close; clc;

x0 = [3; 1];
fun_rJ = @func_rJ_exe;
x_star = [0; 0];
%% Newton
alpha = 1;
[~, stat1] = newton(alpha, fun_rJ, x0);

%% Gauss-Newton
flag_line = false;
[~, stat2] = GaussNewton_line(fun_rJ, x0, flag_line);


%% LM
[~, stat3] = Levenberg_Marquardt(fun_rJ, x0);

%% LM_yq
tau = 1;
[~, stat4] = Levenberg_Marquardt_yq(fun_rJ, x0, tau);

%% LM_yq_new
fun_rJ = @func_rJ_exe_new;
tau = 1e-16;
[~, stat5] = Levenberg_Marquardt_yq(fun_rJ, x0, tau);
%% plots

% e_k
figure;
for i = 1:5
    s = eval('stat'+string(i));
    X = s.F;
    for j = 1:length(X)
        X(j) = norm(s.X(:, j) - x_star, 2);
    end
    if i==2 | i==5
        plot(1:length(X), X, '--', 'linewidth', 2);
    else
        plot(1:length(X), X, '-', 'linewidth', 2);
    end
    hold on;
end
xlabel('iteration');
ylabel('e_k');
legend('Newton', 'Gauss-Newton', 'LM', 'LM_{yq}', 'modified-LM_{yq}');
hold off;

%%
figure;
for i = 1:5
    s = eval('stat'+string(i));
    X = s.F;
    if i==2 || i==5
        plot(1:length(X), X, '--', 'linewidth', 2);
    else
        plot(1:length(X), X, '-', 'linewidth', 2);
    end
    hold on;
end
xlabel('iteration');
ylabel('f(x_k)');
legend('Newton', 'Gauss-Newton', 'LM', 'LM_{yq}', 'modified-LM_{yq}');
hold off;

%%
% convergence rate
figure;
for i = 1:5
    s = eval('stat'+string(i));
    X = s.F;
    for j = 1:length(X)
        X(j) = norm(s.X(:, j) - x_star, 2);
    end
    Y = X;
    Y(end) = [];
    for j = 2:length(X)
        Y(j) = X(j) / X(j-1);
    end
    if i == 5
       for j = 2:length(X)
        Y(j) = X(j) / X(j-1)^2;
       end 
    end
    if i==2 || i==5
        plot(1:length(Y), Y, '--', 'linewidth', 2);
    else
        plot(1:length(Y), Y, '-', 'linewidth', 2);
    end
    hold on;
end
xlabel('iteration');
ylabel('e_{k+1}/e_k');
legend('Newton', 'Gauss-Newton', 'LM', 'LM_{yq}', 'modified-LM_{yq}(e_{k+1}/{e_k}^2)');
hold off;
