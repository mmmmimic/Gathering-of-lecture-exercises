%% Minimizers of univariate and multivariate problems
clear all; close all; clc;
%%
% f(x) = 3/2*(x1^2+x2^2)+(1+a)*x1*x2-(x1+x2)+b
% df(x) = [3*x1+(1+a)*x2-1;
%           3*x2+(1+a)*x1-1]
% Hession = [3,1+a;
%            1+a,3]
% a和b取何值令f(x)有且只有一个最优解？
% x1 = x2 = (2-a)/(9+(1+a)^2)
% eigen value is (2-a) or (4+a)
% -4<=a<=2

%%
% f(x) = 1/2*x^T*Q*x-b^T*x
% dx = Q*x-b^T = 0
% x* = solve(Qx=b)

%%
% min(b^T*x)
% df(x) = b inequal 0
% doesn't meet the optimization necessary condition

%%
