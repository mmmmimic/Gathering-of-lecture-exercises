function [alpha,fnew,dfnew] = backtracking(fundfun,x,p,f,df,varargin)

rho = 0.5;
c = 0.1;

alpha = 1;
[fnew,dfnew] = feval(fundfun,x+alpha*p,varargin{:});
while fnew > (f+c*alpha*(df'*p))
    alpha = rho*alpha;
    [fnew,dfnew] = feval(fundfun,x+alpha*p,varargin{:});
end