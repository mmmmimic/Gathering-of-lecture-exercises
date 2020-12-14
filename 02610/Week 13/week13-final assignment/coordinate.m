function [x,stat]=coordinate(fundfun,x0,gamma0,maxItr,tolerance,varargin)
    % Solver settings and info
    maxit = maxItr;
    tol = tolerance;
    stat.converged = false; % converged
    stat.nfun = 0; % number of function calls
    stat.iter = 0; % number of iterations
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
    n = length(x0);
    E = eye(n);
    while ~converged && (it < maxit)
        it = it+1;
        state = 0;
        % Find the possible descent direction
        for i = -n:n
            if i == 0
                continue
            end
            e = E(:,abs(i))*sign(i);
            fnew = feval(fundfun,x + gamma*e,varargin{:});
            if fnew < f
               state = 1;
               f = fnew;
               x = x + gamma*e;
            end
        end
        % Update iterate
        if state == 0
            gamma = gamma/2;
        end
        converged = (gamma <= tol);
        % Store data for plotting
        stat.X = [stat.X x];
        stat.F = [stat.F f];
    end
    % Prepare return data
    if ~converged
        x = [];
    end
    stat.converged = converged;
    stat.iter = it;
end