function [x,stat] = steepestdescent_line(fundfun,x0,varargin)

% Solver settings and info
maxit = 2000;
tol   = 1.0e-6;

rho = 0.5;
c = 0.1; 


stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations
stat.alpha = [];

% Initial iteration
x = x0;
it = 0;
[f,df] = feval(fundfun,x,varargin{:});
norm_df = norm(df);
converged = (norm_df <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.normdF = norm_df;

% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    
    p = - df/norm(df,2);
    
    % backtracking line search
    alpha = 1;
    fnew = feval(fundfun,x+alpha*p,varargin{:});
    while fnew > (f+c*alpha*(df'*p))
        alpha = rho*alpha;
        fnew = feval(fundfun,x+alpha*p,varargin{:});
    end
    stat.alpha = [stat.alpha alpha];

    x = x+alpha*p;
    
    [f,df] = feval(fundfun,x,varargin{:});
    norm_df = norm(df);
    converged = (norm_df <= tol);
    stat.nfun = stat.nfun+1;
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.F  = [stat.F f];   
    stat.normdF = [stat.normdF norm_df];
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
