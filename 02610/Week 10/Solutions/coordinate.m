function [x,stat]=coordinate(fundfun,x0,gamma0,varargin)

% Solver settings and info
maxit = 10000; %100*length(x0);
tol   = 1.0e-6;
n = length(x0);
I = eye(n);

stat.converged = false;         % converged
stat.nfun = 0;                  % number of function calls
stat.iter = 0;                  % number of iterations

% Initial iteration
gamma = gamma0;
x = x0;
it = 0;
f = feval(fundfun,x,varargin{:});
stat.nfun = 1;
converged = false;

% Store data for plotting
stat.X = x;
stat.F = f;

% Main loop of coordinate search
while ~converged && (it < maxit)
    it = it+1;
    
    D = [repmat(x,1,n)+gamma*I, repmat(x,1,n)-gamma*I];
    fD = zeros(size(D,2),1);
    for ii = 1:2*n
        fD(ii) = feval(fundfun,D(:,ii),varargin{:});
    end
    stat.nfun = stat.nfun+2*n;
    
    [ft,ind] = min(fD);
    if ft < f
        x = D(:,ind);
        f = ft;
    else
        gamma = gamma/2;
    end
    
    converged = (gamma <= tol);
    
    % Store data for plotting
    stat.X  = [stat.X  x];
    stat.F  = [stat.F f];
end

% Prepare return data
if ~converged
     x = []; 
end
stat.converged = converged;
stat.iter = it;
