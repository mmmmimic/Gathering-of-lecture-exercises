function [x,stat] = steepestdescent_line(alpha,fundfun,x0,rho,c,varargin)
% Solver settings and info
maxit = 100*length(x0);
tol = 1.0e-10;
stat.converged = false; % converged
stat.nfun = 0; % number of function calls
stat.iter = 0; % number of iterations
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
stat.alpha = [];
% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    % Steepest descent step
    % TODO -- Insert code between the lines
    % ================================================
    % backtracking line search
    p = - df/norm(df,2);
    fnew = feval(fundfun,x+alpha*p,varargin{:});
    while fnew > (f+c*alpha*(df'*c))
        alpha = rho*alpha;
        fnew = feval(fundfun,x+alpha*p,varargin{:});
    end
    stat.alpha = [stat.alpha alpha];

    x = x+alpha*p;
    % ================================================
    [f,df] = feval(fundfun,x,varargin{:});
    converged = (norm(df,'inf') <= tol);
    stat.nfun = stat.nfun+1;
    % Store data for plotting
    stat.X = [stat.X x];
    stat.F = [stat.F f];
    stat.dF = [stat.dF df];
end
% Prepare return data
if ~converged
    x = [];
end
stat.converged = converged;
stat.iter = it;
