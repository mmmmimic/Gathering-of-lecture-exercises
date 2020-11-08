clear all, close all,

t = [-1.5; -0.5; 0.5; 1.5; 2.5];
y = [0.80; 1.23; 1.15; 1.48; 2.17];

A = [t, ones(size(t))];
x2 = linearLSQ(A,y);

r2 = A*x2-y;
norm_r2 = norm(r2)

figure, plot(t,y,'r*',t,x2(1)*t+x2(2),'b')

%% l-inf regression

f = [zeros(size(x2));1];
B = [-A, -ones(size(A,1),1); A, -ones(size(A,1),1)];
b = [-y; y];

xt = linprog(f,B,b);
xinf = xt(1:2)

rinf = A*xinf-y;
norm_rinf = norm(rinf,inf)

hold on, plot(t,xinf(1)*t+xinf(2),'g')

%% l1 regression

f = [zeros(size(x2)); ones(size(t))];
B = [-A, -eye(size(A,1)); A, -eye(size(A,1))];
b = [-y; y];

xt = linprog(f,B,b);
x1 = xt(1:2)

r1 = A*x1-y;
norm_r1 = norm(r1,1)

plot(t,x1(1)*t+x1(2),'k'), hold off
legend('data','l_2','l_\infty','l_1')

%%
%% with outlier

t = [-1.5; -0.5; 0.5; 1.5; 2.5];
y = [0.80; 1.23; 1.15; 1.48; 4];

A = [t, ones(size(t))];
x2o = linearLSQ(A,y);

hold on, plot(t,y,'bo',t,x2o(1)*t+x2o(2),'b--')

%% l-inf regression

f = [zeros(size(x2));1];
B = [-A, -ones(size(A,1),1); A, -ones(size(A,1),1)];
b = [-y; y];

xt = linprog(f,B,b);
xinfo = xt(1:2)

plot(t,xinfo(1)*t+xinfo(2),'g--')

%% l1 regression

f = [zeros(size(x2)); ones(size(t))];
B = [-A, -eye(size(A,1)); A, -eye(size(A,1))];
b = [-y; y];

xt = linprog(f,B,b);
x1o = xt(1:2)


plot(t,x1o(1)*t+x1o(2),'k--'), hold off
legend('data','l_2','l_\infty','l_1','new data','l_2','l_\infty','l_1')
