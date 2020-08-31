function y = myFun(x)
assert(size(x)<=1);
assert(min(x)>0);
y = x - log(x);
end