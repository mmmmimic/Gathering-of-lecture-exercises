function [x,stat] = BFGSmethod_line(H,maxit,fundfun,x0,varargin)

% Solver settings and info
%maxit = 100*length(x0);
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
[f,df] = feval(fundfun,x,varargin{:});
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;

I = eye(length(x));

% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    
    p = - H*df;
    
    % backtracking line search
    alpha = 1;
    fnew = feval(fundfun,x+alpha*p,varargin{:});
    while fnew > (f+c*alpha*(df'*p))
        alpha = rho*alpha;
        fnew = feval(fundfun,x+alpha*p,varargin{:});
    end
    stat.alpha = [stat.alpha alpha];

    xnew = x+alpha*p;
    [fnew,dfnew] = feval(fundfun,xnew,varargin{:});  
    
    s = xnew-x;
    y = dfnew-df;
    rhok = 1/(y'*s);
    H = (I-rhok*s*y')*H*(I-rhok*y*s')+rhok*s*s';
    
    x = xnew;
    f = fnew;
    df = dfnew;
    
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
