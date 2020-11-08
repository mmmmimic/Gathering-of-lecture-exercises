function [x,stat] = variable_projection(fun_All, a0,varargin)

% Solver settings and info
maxit = 100*length(a0);
tol   = 1.0e-10;

stat.converged = false;         % converged
stat.iter = 0;                  % number of iterations

% Initial iteration
a = a0;
it = 0;

[ra,Ja,ca]=feval(fun_All,a,varargin{:});
f = norm(ra,2)^2/2;
df = Ja'*ra;

converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Initial lambda
lambda = norm(Ja'*Ja,2);

% Store data for plotting
stat.X = [ca;a];
stat.F = f;
stat.dF = df;

% Main loop of variable projection method
while ~converged && (it < maxit)
    it = it+1;
    
    % Calculate the search direction by solving a linear LSQ problem    
    A = [Ja;sqrt(lambda)*eye(length(a))];
    b = [-ra;zeros(size(a))];
    p = linearLSQ(A,b);
    
    % Update the iterate, Jacobian, residual, f, df
    a_new = a+p;
    
    % Update the Lagrange parameter lambda
    [ra_new,Ja_new,ca_new]=feval(fun_All,a_new,varargin{:});
    f_new = norm(ra_new,2)^2/2;

    rho = (f-f_new)/(0.5*(p'*(lambda*p-Ja'*ra)));
    if rho > 0.75
        lambda = lambda/3;
    elseif rho < 0.25
        lambda = 2*lambda;
    end
    
    % Accept or reject x_new
    if rho > 0 
        a = a_new;
        ra = ra_new;
        f = f_new; 
        Ja = Ja_new;
        ca = ca_new;
        df = Ja'*ra;
    end

    converged = (norm(df,'inf') <= tol);
    stat.nfun = stat.nfun+1;
    
    % Store data for plotting
    x = [ca;a];
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
