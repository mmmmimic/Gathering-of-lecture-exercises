function [x,stat] = newton(alpha,fundfun, x0, varargin)

% Solver settings and info
maxit = 2000;
tol   = 1.0e-10;

stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations

% Initial iteration
x = x0;
it = 0;
ek = norm(x0-[1;1], 2);
[f,df, d2f] = feval(fundfun,x,varargin{:});
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;
stat.dX = ek;

% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    
    p = - d2f\df;
    x = x+alpha*p;
    
    [f,df, d2f] = feval(fundfun,x,varargin{:});
    converged = (norm(df,'inf') <= tol);
    stat.nfun = stat.nfun+1;
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.F  = [stat.F f];
    stat.dF = [stat.dF df];
    stat.dX = [stat.dX norm(x-[1;1], 2)];
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
