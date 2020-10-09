function [f, df, d2f] = func2(x)
x1 = x(1);
x2 = x(2);
f = x1^4/4-x1^2+2*x1+(x2-1)^2;
df = [x1^3-2*x1+2;2*x2-2];
d2f = [3*x1^2-2,0;0,2];
end

