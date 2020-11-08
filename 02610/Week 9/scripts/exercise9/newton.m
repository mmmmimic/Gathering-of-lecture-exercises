function [x,stat] = newton(alpha, fundfun, x0, varargin)

% Solver settings and info
maxit = 100*length(x0);
tol   = 1.0e-10;

stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations

% Initial iteration
x = x0;
it = 0;
[r, J] = feval(fundfun,x,varargin{:});
converged = (norm(r, 'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = 0.5 * norm(r, 2)^2;
stat.X1 = x(1);
stat.X2 = x(2);
stat.J = det(J);

% Main loop of steepest descent
while ~converged && (it < maxit)
    it = it+1;
    
    p = - J \ r;
    x = x + alpha*p;
    
    [r, J] = feval(fundfun,x,varargin{:});
    converged = (norm(r, 'inf') <= tol);
    stat.nfun = stat.nfun+1;
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.F  = [stat.F 0.5 * norm(r, 2)^2];
    stat.X1 = [stat.X1, x(1)];
    stat.X2 = [stat.X2, x(2)];
    stat.J = [stat.J, det(J)];
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
