%clc, clear all, close all,

n = 10;
A=gallery('poisson',n);
b = ones(n^2,1);
x0 = zeros(n^2,1);

m = 10;
[x,stat] = LBFGSmethod(@funExe2,x0,m,A,b);

figure,
semilogy(stat.normdF), title('Residual'),

disp('Not as good as CG but better than SD.')


