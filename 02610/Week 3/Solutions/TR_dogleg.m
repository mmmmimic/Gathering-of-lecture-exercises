function [x,stat] = TR_dogleg(eta,fundfun,x0,varargin)

% Solver settings and info
maxit = 100*length(x0);
tol   = 1.0e-10;
delta_hat = 10;  % upper bound of delta
delta = 1;   % initial delta

% define the function m
m = @(p,f,df,d2f)(f+df'*p+0.5*(p'*d2f*p));

stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations

% Initial iteration
x = x0;
it = 0;
[f,df,d2f] = feval(fundfun,x,varargin{:});
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;

% Main loop of trust-region
while ~converged && (it < maxit)
    it = it+1;
    
    % Obtain the search direction p
    p = dogleg(df,d2f,delta);
    norm_p = norm(p);
    
    xnew = x+p;
    [fnew,dfnew,d2fnew] = feval(fundfun,xnew,varargin{:});
    
    rho = (f-fnew)/(m(zeros(size(p)),f,df,d2f)-m(p,f,df,d2f));
    if rho < 0.25
        delta = 0.25*delta;
    elseif rho > 0.75 && abs(norm_p-delta) < 1e-8
        delta = min(2*delta, delta_hat);
    end
    
    if rho > eta
        x = xnew;
        f = fnew;
        df = dfnew;
        d2f = d2fnew;
    end
    
    converged = (norm(df,'inf') <= tol);
    stat.nfun = stat.nfun+1;
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.F  = [stat.F f];
    stat.dF = [stat.dF df];
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
