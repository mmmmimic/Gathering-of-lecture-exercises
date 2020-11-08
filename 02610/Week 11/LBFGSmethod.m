function [x,stat] = LBFGSmethod(fundfun,x0,m,varargin)

% Solver settings and info
maxit = 500;
tol   = 1.0e-6;

n = length(x0);
Sm = zeros(n,m);
Ym = zeros(n,m);
 
stat.converged = false;         % converged
stat.iter = 0;                  % number of iterations
stat.alpha = [];

% Initial iteration
x = x0;
[f,df] = feval(fundfun,x,varargin{:});
norm_df = norm(df);
converged = (norm_df <= tol);

% Store data for plotting
stat.X = x;
stat.F = f;
stat.normdF = norm_df;

if converged
    return;
end

% 1st iteration
[alpha,fnew,dfnew] = backtracking(fundfun,x,-df,f,df,varargin{:});
xnew = x-alpha*df;
norm_df = norm(dfnew);
converged = (norm_df <= tol);

% Store data for plotting
stat.X  = [stat.X  xnew];
stat.F  = [stat.F fnew];
stat.normdF = [stat.normdF norm_df];

if converged
    x = xnew;
    stat.iter = 1;
    return;
end

it = 0;
% Main loop of L-BFGS
while ~converged && (it < maxit)
    it = it+1;
    
    s = xnew-x;
    y = dfnew-df;

    % Compute pk
    gamma = (s'*y)/(y'*y);
    if it<=m
        Sm(:,it) = s;
        Ym(:,it) = y;
        p = -getHg_lbfgs(dfnew,Sm(:,1:it),Ym(:,1:it),gamma);
    else
        Sm(:,1:(m-1)) = Sm(:,2:m);
        Ym(:,1:(m-1)) = Ym(:,2:m);
        Sm(:,m) = s;
        Ym(:,m) = y;
        p = -getHg_lbfgs(dfnew,Sm,Ym,gamma);
    end
    
    % update x
    x = xnew;
    f = fnew;
    df = dfnew;
    [alpha,fnew,dfnew] = backtracking(fundfun,x,p,f,df,varargin{:});
    stat.alpha = [stat.alpha alpha];

    xnew = x+alpha*p;
    
    norm_df = norm(dfnew);
    converged = (norm_df <= tol);

    % Store data for plotting
    stat.X  = [stat.X  xnew];
    stat.F  = [stat.F fnew];
    stat.normdF = [stat.normdF norm_df];
    
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it+1;
