function [x,stat] = BFGSmethod_line(iniAlpha,rho,c,fundfun,x0, Hini,maxItr,tolerance)
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
    Hitr = Hini;
    shape = size(Hitr);
    mtxIt = eye(shape(1));
    x = x0;
    it = 0;
    [f,df,d2f] = feval(fundfun, x);
    converged = (norm(df,'inf') <= tol); % max(abs(x))
    stat.nfun = 1;
    
    % Store data for plotting
    stat.X = x;
    stat.F = f;
    stat.dF = norm(df);
    stat.Alpha = iniAlpha;
    
    % Main loop for steepest descent
    while ~converged && (it < maxit)
        it = it + 1;
        
        % Steepest descent step
        p = -Hitr*df;
        
        % Determine alpha by backtracking
        alpha = iniAlpha;
        xm = x + alpha*p;
        [fm,dfm,d2fm] = feval(fundfun, xm);
        while fm > f + c*alpha*df'*p
            alpha = rho*alpha;
            xm = x + alpha*p;
            [fm,dfm,d2fm] = feval(fundfun, xm);
        end
        
        % Converge condition check
        yItr = dfm - df;
        sItr = alpha*p;
        rhoItr = 1/(yItr'*sItr);
        Hitr = Hitr + (mtxIt - rhoItr*sItr*yItr')*Hitr*(mtxIt - rhoItr*sItr*yItr') + rhoItr*(sItr*sItr');
        f = fm;
        df = dfm;
        x = xm;
        converged = (norm(df,'inf') <= tol);
        stat.nfun = stat.nfun+1;
        stat.X = [stat.X x];
        stat.F = [stat.F f];
        stat.dF = [stat.dF norm(df)];
        stat.Alpha = [stat.Alpha alpha];
    end
    
    % Prepare return data
    if ~converged
        x = [];
    end
    stat.converged = converged;
    stat.iter = it;
end