function f=rosenbrock(x)

tmp = x(2)-x(1)^2;
f = 100*tmp^2+(1-x(1))^2;
