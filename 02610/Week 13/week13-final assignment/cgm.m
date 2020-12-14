function [x,stat]=cgm(A,b,x0)
% Solver settings and info
maxit = 100;
tol = 1.0e-6;
stat.converged = false; % converged
stat.iter = 0; % number of iterations
% Initial iteration
x = x0;
it = 0;
r = b-A*x;
p = r;
norm_r = norm(r);
converged = false;
% Store data for plotting
stat.X = x;
stat.resd = norm_r; % norm of residuals
% Main loop of conjugate gradient
while ~converged && (it < maxit)
    it = it+1;
    % TODO -- implement main loop of CG method
    alpha = r'*r/(p'*A*p);
    x = x + alpha*p;
    rn = r - alpha*A*p;
    beta = rn'*rn/(r'*r);
    p = rn + beta*p;
    r = rn;
    norm_r = norm(r);
    % Set the stopping rule
    converged = (norm_r <= tol);
    % Store data for plotting
    stat.X = [stat.X x];
    stat.resd = [stat.resd norm_r];
end
% Prepare return data
if ~converged
    x = [];
end
stat.converged = converged;
stat.iter = it;
end