function [f, df, ddf] = rosenbrock(x)
x1 = x(1);
x2 = x(2);
% rosenbrock
f = 100*(x2-x1.^2).^2+(1-x1).^2;

% gradient
df = [-400*(x2-x1^2)*x1-2*(1-x1);
    200*(x2-x1^2)];

% Hession
ddf = [-400*x2+1200*x1.^2+2, -400*x1;
    -400*x1,200];
end