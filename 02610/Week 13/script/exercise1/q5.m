%% Question 5
clear, close all, clc;
%%
x0 = [1;4];
rho = 0.5;
c = 0.1;
H = eye(2);
gamma0 = 1;
alpha = 1;

X = [];
len = [];
%% steepest descent method
[~,stat] = steepestdescent_line(@foo, x0, rho, c);
x = show_track(stat);
X = [X, x];
len = [len, length(x)];
%% Newton's method
[~,stat] = newton(alpha, @foo, x0);
x = show_track(stat);
X = [X, x];
len = [len, length(x)];
%% BFGS method
[~,stat] = BFGSmethod_line(H, @foo, x0, rho, c);
x = show_track(stat);
X = [X, x];
len = [len, length(x)];
%% coordinate search method
[~,stat]=coordinate(@foo, x0, gamma0);
x = show_track(stat);
X = [X, x];
len = [len, length(x)];
%% Nonlinear conjugate gradient method
[~,stat] = nonlinear_conj_line(@foo,x0,rho,c);
x = show_track(stat);
X = [X, x];
len = [len, length(x)];
%% transfer data to python to generate latex script
save 'data.mat' X;

save 'len.mat' len;
%% new starting point
x0 = [-3;-2];
% Newton's method
[~,stat] = newton(alpha, @foo, x0);
newton_x = show_track(stat);

%% new starting point
x0 = [0;0.5];
% Newton's method
[~,stat] = newton(alpha, @foo, x0);
newton_x = show_track(stat);
%%
function point = show_track(stat)
x_star = [1;1];
% Save and output the first 8 iterates
if size(stat.X, 2)>=8
    point = stat.X(:, 1:9);
else
    point = stat.X;
end
q3;
hold on;
plot(point(1, :), point(2, :), 'r-', 'Linewidth',2);
hold on;
plot(point(1, :), point(2, :), 'g.', 'Linewidth',5);
hold off;

figure;
F = stat.F;
X = F;
DF = F;
for i = 1:length(stat.X)
    X(i) = norm(stat.X(i) - x_star, 2);
    try
        DF(i) = norm(stat.dF(i), 2);
    catch
        DF = [];
    end
end
subplot(3,1,1);
% draw ||x_k - x^*||
semilogy([1:length(F)], X);
xlabel('iteration');
ylabel('{||x_k - x^*||}_2')

subplot(3,1,2);
% draw f(x_k)
semilogy([1:length(F)], F);
xlabel('iteration');
ylabel('f(x_k)')

subplot(3,1,3);
% draw ||df(x_k)||
semilogy([1:length(DF)], DF);
xlabel('iteration');
ylabel('{||df(x_k)||}_2')
end