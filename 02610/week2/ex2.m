clear all; close all; clc;
%% ex2.3.1 (a)
x0 = 0.1;
alpha = 0.1;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-x);
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%% ex2.3.1 (b)
x0 = 5;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-x);
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%%
x0 = 0.5;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-x);
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%% ex2.3.1 (c)
x0 = 5;
alpha = 0.5;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-x);
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%% ex2.3.2
x0 = 0.1;
alpha = 1;
[x,stat] = newton(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-x);
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%% ex2.3.3
x0 = 0.01;
alpha = 0.01;
[x,stat] = steepestdescent(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-xk(end));
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%%
x0 = 0.01;
alpha = 0.01;
[x,stat] = newton(alpha,@PenFun1,x0,1);
xk = stat.X;
e = abs(xk-xk(end));
df = abs(stat.dF);
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%%
% steepest descent converages faster here. 
% but when it is approaching the minimizer, the speed becomes very slow
% newton method doesn't have this problem, but it is more computational
% expensive. And in the early stage, it's slow. 

%% ex2.4.1
[f,df,ddf] = MyFun([5,10]);
%% ex2.4.2
x0 = [10;1];
alpha = 0.05;
[x,stat] = steepestdescent(alpha,@MyFun,x0);
%[x,stat] = newton(alpha,@MyFun,x0);
xk = stat.X;
e = [];
df = [];
for i = 1:stat.iter+1
   e = [e,norm(xk(i)-xk(end))];
   df = [df,norm(stat.dF(i))]; 
end
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%% ex2.4.3
x0 = [10;1];
alpha = 1;
[x,stat] = newton(alpha,@MyFun,x0);
xk = stat.X;
e = [];
df = [];
for i = 1:stat.iter+1
   e = [e,norm(xk(i)-xk(end))];
   df = [df,norm(stat.dF(i))]; 
end
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%%
% In this question, Newton method is appearantly much faster than Steepest
% descent method, even though it needs to compute the Hession matrix. 

%% ex2.5.1
[x,stat] = steepestdescent_line(1,@MyFun,x0,0.5,0.1);
xk = stat.X;
e = [];
df = [];
for i = 1:stat.iter+1
   e = [e,norm(xk(i)-xk(end))];
   df = [df,norm(stat.dF(i))]; 
end
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
%% ex2.5.2
[x,stat] = newton_line(1,@MyFun,x0,0.5,0.1);
xk = stat.X;
e = [];
df = [];
for i = 1:stat.iter+1
   e = [e,norm(xk(i)-xk(end))];
   df = [df,norm(stat.dF(i))]; 
end
f = stat.F;
k = [0:stat.iter]';
T = table(k, xk', e', df', f');
figure;
subplot(3,1,1);
plot(k,df);
title('df')
subplot(3,1,2);
plot(k,f);
title('f')
subplot(3,1,3);
plot(k,e);
title('e');
% Now the result is much better
