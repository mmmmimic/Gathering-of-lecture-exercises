%% Convexity
clear all; close all; clc;
%%
% assume 2 points in the set, Ax1<=b, Ax2<=b
% a*x1+(1-a)*x2
%a*A*x1+(1-a)*A*x2 <= a*b+(1-a)*b=b
% thus the point a*x1+(1-a)*x2 is in the set
% proof end

%%
f1 = @(x) x.^2+x+1;
df1 = @(x) 2*x+1;
ddf1 = @(x) ones(1,size(x,2))*2;
f2 = @(x) -x.^2+x+1;
df2 = @(x) -2*x+1;
ddf2 = @(x) -2*ones(1,size(x,2));
f3 = @(x) x.^3-5*x.^2+x+1;
df3 = @(x) 3*x.^2 - 10*x +1;
ddf3 = @(x) 6*x - 10;
f4 = @(x) x.^4+x.^3-10*x.^2-x+1;
df4 = @(x) 4*x.^3+3*x.^2-20*x-1;
ddf4 = @(x) 12*x.^2+6*x-20;
x1 = [-2:0.01:2];
x2 = [-4:0.1:4];

figure(1);
subplot(3,1,1);
plot(x1, f1(x1));
title('function');
subplot(3,1,2);
plot(x1, df1(x1));
title('1-st order derivation');
subplot(3,1,3);
plot(x1, ddf1(x1));
title('2-nd order derivation');

figure(2);
subplot(3,1,1);
plot(x1, f2(x1));
title('function');
subplot(3,1,2);
plot(x1, df2(x1));
title('1-st order derivation');
subplot(3,1,3);
plot(x1, ddf2(x1));
title('2-nd order derivation');

figure(3);
subplot(3,1,1);
plot(x2, f3(x2));
title('function');
subplot(3,1,2);
plot(x2, df3(x2));
title('1-st order derivation');
subplot(3,1,3);
plot(x2, ddf3(x2));
title('2-nd order derivation');

figure(4);
subplot(3,1,1);
plot(x2, f4(x2));
title('function');
subplot(3,1,2);
plot(x2, df4(x2));
title('1-st order derivation');
subplot(3,1,3);
plot(x2, ddf4(x2));
title('2-nd order derivation');

%%
% function 1

%%
figure(1);
subplot(3,1,1);
plot(x1, f1(x1));
title('function');
subplot(3,1,2);
plot(x1, df1(x1));
title('1-st order derivation');
subplot(3,1,3);
plot(x1, ddf1(x1));
title('2-nd order derivation');
% local
% minimizer = -0.5
% maximizer = None
% global
% minimizer = -0.5
% maximizer = 2
% conditions
% 1-st order derivation: 0
% 2-nd order derivation: >0
%%
figure(2);
subplot(3,1,1);
plot(x1, f2(x1));
title('function');
subplot(3,1,2);
plot(x1, df2(x1));
title('1-st order derivation');
subplot(3,1,3);
plot(x1, ddf2(x1));
title('2-nd order derivation');
% local
% minimizer = -2
% maximizer = None
% global
% minimizer = -2
% maximizer = 0.5
% conditions
% 1-st order derivation: 0
% 2-nd order derivation: <0
%%
figure(3);
subplot(3,1,1);
plot(x2, f3(x2));
title('function');
subplot(3,1,2);
plot(x2, df3(x2));
title('1-st order derivation');
subplot(3,1,3);
plot(x2, ddf3(x2));
title('2-nd order derivation');
% local
% minimizer = 3
% maximizer = 0
% global
% minimizer = -4
% maximizer = 0
% conditions
% 1-st order derivation: 0
% 2-nd order derivation: >0 || <0
%%
figure(4);
subplot(3,1,1);
plot(x2, f4(x2));
title('function');
subplot(3,1,2);
plot(x2, df4(x2));
title('1-st order derivation');
subplot(3,1,3);
plot(x2, ddf4(x2));
title('2-nd order derivation');
% local
% minimizer = -2.5,2
% maximizer = 0
% global
% minimizer = -2.5
% maximizer = 4
% conditions
% 1-st order derivation: 0
% 2-nd order derivation: >0 / <0
%%
% f(x) = 2*x1^2-2*x1*x2+1/2*x2^2+3*x1-x2
% f(x) = x1*(2*x1-x2+3)+x2*(1/2*x2-x1-1)
% f(x) = [x1 x2]*[2x1-x2+3;x2/2-x1+1]
% f(x) = [x1 x2]*[2 -1 3;-1 1/2 -1]*[x1;x2;1]
% df(x) = [4x1-2x2+3;
%           x2-2x1-1]
% Hession = [4,-2;-2,1]
% det(Hession) = 0, it's singular
% eigen value = 0 or 5
% Hession matrix is semi-definite
% then f(x) is a convex function

%%
% f(global minimizers) must be equal, and they must be on a horizontal line
% assume f({x*}) = b
% thus a*f(x1)+(1-a)*f(x2) = b = f(somewhere between x1 and x2) = f(a*x1+(1-a)*x2)
% thus {x*} is a convex set

%%
% determine a convex problem
% three conditions
% the objective function is convex
% the equal constrainit function is linear
% the inequal constraint function is concave
% f(x) = x*log(x) is convex (negative entrophy)
% x is convex, thus this is not a convex problem

