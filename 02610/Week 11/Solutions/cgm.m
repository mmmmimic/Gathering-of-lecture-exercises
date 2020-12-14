function [x,stat]=cgm(A,b,x0)

% Solver settings and info
maxit = 100; 
tol   = 1.0e-6;

stat.converged = false;         % converged
stat.iter = 0;                  % number of iterations

% Initial iteration
x = x0;
it = 0;
r = b-A*x;
p = r;
norm_r = norm(r);
converged = false;

% Store data for plotting
stat.X = x;
stat.resd = norm_r;

% Main loop of conjugate gradient
while ~converged && (it < maxit)
    it = it+1;
    
    Ap = A*p;
    alpha = norm_r^2/(p'*Ap);
    x = x+alpha*p;
    r = r-alpha*Ap;
    norm_r_New = norm(r);
    beta = (norm_r_New/norm_r)^2;
    p = r+beta*p;
    
    norm_r = norm_r_New;    
    converged = (norm_r <= tol);
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.resd  = [stat.resd norm_r];
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
