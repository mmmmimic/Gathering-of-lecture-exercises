function A = genA(tau)

if tau<=0 || tau>=1
    error('tau need be in (0,1)!');
end

n = 500;
A = speye(n);

rng(100);
R = rand(n)*2-1;
R = R-diag(diag(R));

A = A+R;
A = (A+A')/2;

Ind = abs(A)>tau;
A(Ind) = 0;
A = A+speye(n);
