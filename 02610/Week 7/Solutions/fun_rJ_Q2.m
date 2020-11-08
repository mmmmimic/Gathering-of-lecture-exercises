function [r,J]=fun_rJ_Q2(x,lambda)

r=[x+1; lambda*x^2+x-1];
J=[1; 2*lambda*x+1];