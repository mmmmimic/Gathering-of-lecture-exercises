%%  Function plots and contour plots in Matlab
clear all; close all; clc;
%% 1.1: use function
step = 0.01;
x = [step:step:2];
y = myFun(x);
figure;
plot(x,y);

%% 1.1: use function handle
myFunc = @(x) x-log(x);
y = myFunc(x);
figure;
plot(x,y);

%% 1.1: derivatives
myFuncd = @(x) 1-x.^-1;
myFuncdd = @(x) x.^-2;
figure;
subplot(1,3,1);
plot(x,y);
title('function');
subplot(1,3,2);
plot(x,myFuncd(x));
title('1st-order derivatons');
subplot(1,3,3);
plot(x,myFuncdd(x));
title('2nd-order derivations');
%% 1.1 is the function convex?
figure;
plot(x,y);
ylim([0.5,5]);
title('function');
% from the figure it's easy to learn that the function is convex

%% 1.1 localize the minimizer
% when 1st-order derivation is 0
% 1-1/x = 0, x = 1
% since the 2-nd derivation = 1/(x^2), and x>0
% 2-nd derivation >0 for all x>0
% the minimizer is unique
% another simple reason is, the function is convex
x_star = 1;
disp(myFunc(x_star));

%% 1.2
% hand calculation
% dx1 = 8+2*x1
% dx2 = 12-4*x2
% stationary point (-4, 3)
x1 = [-6:0.01:-2];
x2 = [1:0.01:5];
[X,Y] = meshgrid(x1,x2);
func = @(x1,x2)8*x1+12*x2+x1.^2-2*x2.^2;
y = func(X,Y);
figure;
subplot(1,2,1);
mesh(X,Y,y);
subplot(1,2,2);
v = [-20:20];
[c,h] = contour(x1,x2,y,v,'linewidth',2);
color bar, axis image,
xlabel('x1');
ylabel('x2');
