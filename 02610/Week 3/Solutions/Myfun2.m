function [y, dy, d2y] = Myfun2(x)

f = @(x)(0.5*x(1)^2+5*x(2)^2);
df = @(x)([x(1);10*x(2)]);
d2f = @(x)([1,0;0,10]);

y = f(x);
dy = df(x);
d2y = d2f(x);