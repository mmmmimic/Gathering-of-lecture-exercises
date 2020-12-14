function [x,stat] = nonlinear_conj_line(fundfun, x0, rho, c, varargin)

% Solver settings and info
maxit = 200;
tol   = 1.0e-6;


stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations
stat.alpha = [];

% Initial iteration
x = x0;
it = 0;
[f,df] = feval(fundfun,x,varargin{:});
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;

p = -df;
% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    
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
    converged = (norm(df,'inf') <= tol);
    stat.nfun = stat.nfun+1;
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.F  = [stat.F f];
    stat.dF = [stat.dF df];
    beta = df'*(df-stat.dF(:, it))/(norm(stat.dF(:, it), 2))^2;
    p = - df + beta*p;
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
