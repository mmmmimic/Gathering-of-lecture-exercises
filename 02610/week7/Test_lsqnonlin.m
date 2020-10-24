function Test_lsqnonlin()
clc, clear all, close all,

global t y
load data_exe3

x0 = [3; -3; 3; 3];


options = optimoptions('lsqnonlin','Display','iter');
options.OptimalityTolerance = 1e-10;
options.FunctionTolerance = 1e-10;
options.Jacobian = 'on';

x = lsqnonlin(@fun_rJ_Q4,x0,[],[],options)

plot(t,y,'r.',t,x(1)*exp(-x(3)*t)+x(2)*exp(-x(4)*t),'g--'), hold off



function [r,J]=fun_rJ_Q4(x)
global t y
r=y-x(1)*exp(-x(3)*t)-x(2)*exp(-x(4)*t);
J=[-exp(-x(3)*t), -exp(-x(4)*t), x(1)*t.*exp(-x(3)*t), x(2)*t.*exp(-x(4)*t)];
