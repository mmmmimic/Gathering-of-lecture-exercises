function [x,stat] = newton_line(fundfun,x0,varargin)

% Solver settings and info
maxit = 100*length(x0);
tol   = 1.0e-10;

rho = 0.5;
c = 0.1; 


stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations
stat.alpha = [];


% Initial iteration
x = x0;
it = 0;
[f,df, d2f] = feval(fundfun,x,varargin{:});
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;

% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    
    p = - d2f\df;
    
    % backtracking line search
    alpha = 1;
    fnew = feval(fundfun,x+alpha*p,varargin{:});
    while fnew > (f+c*alpha*(df'*p))
        alpha = rho*alpha;
        fnew = feval(fundfun,x+alpha*p,varargin{:});
    end
    stat.alpha = [stat.alpha alpha];

    x = x+alpha*p;
    
    [f,df,d2f] = feval(fundfun,x,varargin{:});
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
