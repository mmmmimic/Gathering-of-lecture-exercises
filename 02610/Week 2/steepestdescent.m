function [x,stat] = steepestdescent(alpha,fundfun,x0,varargin)
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
% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    % Steepest descent step
    % TODO -- Insert code between the lines
    % ================================================
    p = -df/norm(df,2);
    x = x + alpha*p;
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
