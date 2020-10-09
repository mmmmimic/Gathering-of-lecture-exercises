fundfun = 'func1';
X = [-2,-2,-2;-1,0,1];
rho = 0.5;
c = 0.5;
maxit = 2000;
tol = 1e-5;
for i = 1:3
   x0 = X(:, i); 
   [x,stat] = steepestdescent_line(fundfun,x0, maxit, tol, rho, c);
   disp('For point: ');
   x0
   disp('After iterations: ');
   stat.iter
   disp('The method converged to point: ')
   x
end