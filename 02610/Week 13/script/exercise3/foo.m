function [f, df] = foo(x)
x1 = x(1); x2 = x(2);

f = 1/2*((x1-1).^2+x2.^2);

df = [x1-1; x2];
end