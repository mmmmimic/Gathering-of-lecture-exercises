function [x,stat] = Levenberg_Marquardt(fun_rJ,x0,varargin)

% Solver settings and info
maxit = 100*length(x0);
tol   = 1.0e-10;

stat.converged = false;         % converged
stat.iter = 0;                  % number of iterations

% Initial iteration
x = x0;
it = 0;
[rx,Jx] = feval(fun_rJ,x,varargin{:});
f = norm(rx,2)^2/2;
df = Jx'*rx;
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Initial lambda
lambda = norm(Jx'*Jx,2);
%lambda = 10^(-16)*norm(Jx'*Jx,2);
% nu = 2;  % another way to update lambda

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;

% Main loop of L-M method
while ~converged && (it < maxit)
    it = it+1;
        
    % Calculate the search direction by solving a linear LSQ problem
    A = [Jx;sqrt(lambda)*eye(length(x))];
    b = [-rx;zeros(size(x))];
    p = linearLSQ(A,b);
    
    % Update the iterate, Jacobian, residual, f, df
    x_new = x+p;
    
    % Update the Lagrange parameter lambda
    [rx_new,Jx_new] = feval(fun_rJ,x_new,varargin{:});
    f_new = norm(rx_new,2)^2/2;

    rho = (f-f_new)/(0.5*(p'*(lambda*p-Jx'*rx)));
    if rho > 0.75
        lambda = lambda/3;
    elseif rho < 0.25
        lambda = 2*lambda;
    end
    
    % Accept or reject x_new
    if rho > 0 
        x = x_new;
        rx = rx_new;
        f = f_new; 
        Jx = Jx_new;
        df = Jx'*rx;
        % another way to update lambda
%         lambda = lambda*max(1/3, 1-(2*rho-1)^3);  
%         nu = 2;
%     else
%         lambda = lambda*nu;
%         nu = 2*nu;
    end

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
