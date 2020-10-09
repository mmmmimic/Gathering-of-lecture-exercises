clear;
fundfun = 'func2';
rho = 0.5;
c = 0.5;
maxit = 2000;
tol = 1e-8;
x0 = [-1;0]; 
[~,stat1] = newton(1, fundfun,x0, maxit, tol);
[~,stat2] = newton_line(fundfun,x0, maxit, tol, rho, c);
[~,stat3] = steepestdescent_line(fundfun,x0, maxit, tol, rho, c);

figure,
subplot(3,1,1)
plot(0:stat1.iter, stat1.F, 'linewidth', 1);
ylabel('f(x_k)');
title('Newton')
subplot(3,1,2)
plot(0:stat2.iter, stat2.F, 'linewidth', 1);
ylabel('f(x_k)');
title('Newton line')
subplot(3,1,3)
plot(0:stat3.iter, stat3.F, 'linewidth', 1);
xlabel('iterations');
ylabel('f(x_k)');
title('SD line')

%The  Newton’s  method  converged  much  faster  than  that  with  line  search.   
%The  steepestdescent method didn’t converge.Due to strict prerequisites, 
%the Newton’s method sometimes doesn’t work and even if we addline search, 
%the convergence is much slower than that of steepest descend method.  
%However, ifthe Hession matrix is always definite, the Newton’s method is 
%deemed to be a strong competitorof the best result.  To promise convergence 
%and efficiency, one ought to use the Newton’s methodwith line search, 
%since it works efficiently in most cases.
