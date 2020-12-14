function [f,df] = funExe2(x,A,b)
f = 0.5*(x'*(A*x))-b'*x;
df = A*x-b;