function [x,stat] = GaussNewton_line(fun_rJ,x0,flag_line,varargin)
% Input: flag_line is to control if using backtracking line search.
%        If flag_line=1, we use backtracking line search,
%        otherwise, we use the fixed step length 1.

% Solver settings and info
maxit = 100*length(x0);
tol   = 1.0e-10;

% parameter for backtracking line search
rho = 0.5;  
c = 0.1; 


stat.converged = false;         % converged
stat.iter = 0;                  % number of iterations
stat.alpha = [];

% Initial iteration
x = x0;
it = 0;
[rx,Jx] = feval(fun_rJ,x,varargin{:});
f = norm(rx,2)^2/2;
df = Jx'*rx;
converged = (norm(df,'inf') <= tol);
stat.nfun = 1;

% Store data for plotting
stat.X = x;
stat.F = f;
stat.dF = df;

% Main loop of Gauss-Newton
while ~converged && (it < maxit)
    it = it+1;
    
    % Calculate the search direction by solving a linear LSQ problem
    p = linearLSQ(Jx,-rx);
    
    if flag_line==1
        % backtracking line search
        alpha = 1;
        x_new = x+alpha*p;
        rx_new = feval(fun_rJ,x_new,varargin{:});
        f_new = norm(rx_new,2)^2/2;

        while f_new > (f+c*alpha*(df'*p))
            alpha = rho*alpha;
            x_new = x+alpha*p;
            rx_new = feval(fun_rJ,x_new,varargin{:});
            f_new = norm(rx_new,2)^2/2;    
        end
        stat.alpha = [stat.alpha alpha];
    else
        % fixed alpha
        alpha = 1;
    end

    % Update the iterate, Jacobian, residual, f, df
    x = x+alpha*p;
    
    [rx,Jx] = feval(fun_rJ,x,varargin{:});
    f = norm(rx,2)^2/2;
    df = Jx'*rx;
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
