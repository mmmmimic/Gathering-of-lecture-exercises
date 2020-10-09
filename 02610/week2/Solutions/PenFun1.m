function [f, df, d2f] = PenFun1(x, mu)

if x > 0
    f = x - mu*log(x);
    df = 1.0 - mu./x;
    d2f = mu./(x.*x);
else 
    error(' x need be positive');
end
