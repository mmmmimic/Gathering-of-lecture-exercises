options = optimset('Display','iter');
[x,feval,exitflag,output]=fminsearch(@rosenbrock,[-1.2;1],options);
%[x,feval,exitflag,output]=fminsearch(@rosenbrock,[1.2;1.2],options);