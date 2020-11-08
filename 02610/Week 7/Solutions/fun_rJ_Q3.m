function [r,J]=fun_rJ_Q3(x,t,y)

r=y-x(1)*exp(-x(3)*t)-x(2)*exp(-x(4)*t);
J=[-exp(-x(3)*t), -exp(-x(4)*t), x(1)*t.*exp(-x(3)*t), x(2)*t.*exp(-x(4)*t)];
