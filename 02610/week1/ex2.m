%% Gradient and Hessian of functions
clear all; close all; clc;
%% 2.1
% f(x) = g1*x1+g2*x2+g3*x3
% df(x) = [g1;g2;g3]
% in more genral case
% df(x) = [g1;g2;...;gn]

%% 2.2
% Expand the expression of f(x)
% f(x) = 1/2*(h11*x1^2+h22*x2^2+h33*x3^2+(h12+h21)*x1*x2+(h13+h31)*x1*x3+(h23+h32)*x2*x3)
% df(x) = [ h11*x1+0.5*(h12+h21)*x2+0.5*(h13+h31)*x3
%           h22*x2+0.5*(h12+h21)*x1+0.5*(h23+h32)*x3
%           h33*x3+0.5*(h13+h31)*x1+0.5*(h23+h32)*h2]
% Hession = [h11,0.5*(h12+h21),0.5*(h13+h31)
%           0.5*(h12+h21),h22,0.5*(h23+h32)
%           0.5*(h13+h31),0.5*(h23+h32),h33];

% now assume the matrix is symmteric
% f(x) = 0.5*h11*x1*2+0.5*h22*x2^2+0.5*h33*x3^2+h12*x1*x2+h13*x1*x3+h23*x2*x3
% df(x) = [h11*x1+h12*x2+h13*x3;h22*x2+h12*x1+h23*x3;h33*x3+h13*x1+h23*x2]
% Hession  = [h11,h12,h13;h12,h22,h23;h13,h23,h33]

% df(x) = [h11*x1+h12*x2+...+h1n*xn;h21*x1+h22*x2+...+h2n*xn,...,hn1*x1+hn2*x2+...+hnn*xn]
% Hession = [h11,h12,...,h1n;h21,h22,...,h2n;...;hn1,hn2,...,hnn]
