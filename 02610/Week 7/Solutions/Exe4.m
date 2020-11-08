clc, clear all, close all,

global t y
load data_exe3

fun_f = @(x)y-x(1)*exp(-x(3)*t)-x(2)*exp(-x(4)*t);
x0 = [3; -3; 3; 3];


options = optimoptions('lsqnonlin','Display','iter');
options.Jacobian = 'off';
options.OptimalityTolerance = 1e-10;
options.FunctionTolerance = 1e-10;
options.Algorithm = 'trust-region-reflective';

xopt1 = lsqnonlin(fun_f,x0,[],[],options);

figure, plot(t,y,'r.',t,xopt1(1)*exp(-xopt1(3)*t)+xopt1(2)*exp(-xopt1(4)*t),'b')

%%

options.Algorithm = 'levenberg-marquardt';

xopt2 = lsqnonlin(fun_f,x0,[],[],options);

hold on, plot(t,xopt2(1)*exp(-xopt2(3)*t)+xopt2(2)*exp(-xopt2(4)*t),'k:')

