function [x,stat] = newton_fixedStep(alpha,fundfun,x0,maxItr,tolerance)
% varargin allows the function to take any number of input arguments.

    % Solver settings and info
    if maxItr <= 0
        error('Iteration number must be positive.')
    end
    maxit = maxItr;
    tol = tolerance;
    stat.converged = false;
    stat.nfun = 0;
    stat.iter = 0;
    
    % Initial iteration
    x = x0;
    it = 0;
    [f,df,d2f] = feval(fundfun, x);
    converged = (norm(df,'inf') <= tol); % max(abs(x))
    stat.nfun = 1;
    
    % Store data for plotting
    stat.X = x;
    stat.F = f;
    stat.dF = norm(df);
    
    % Main loop for steepest descent
    while ~converged && (it < maxit)
        it = it + 1;
        % Newton descent step
        p = -pinv(d2f)*df;
        x = x + alpha*p;
        % Converge condition check
        [f,df,d2f] = feval(fundfun, x);
        converged = (norm(df,'inf') <= tol);
        stat.nfun = stat.nfun+1;
        stat.X = [stat.X x];
        stat.F = [stat.F f];
        stat.dF = [stat.dF norm(df)];
    end
    
    % Prepare return data
    if ~converged
        x = [];
    end
    stat.converged = converged;
    stat.iter = it;
end