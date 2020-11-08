t=[-2:1:2]';
y=[0; 15; 25; 50; 110];

A=[ones(size(t)), t, t.^2-2];
disp('The system matrix of the normal equation is')
A'*A
disp('The right-hand side of the normal equation is')
A'*y
%%

disp('Using backslash in Matlab:')
x_b=(A'*A)\(A'*y)

disp('Using QR factorization in Matlab:')
x_qr=linearLSQ(A,y)

%%
r = A*x_qr-y;
tau = norm(r,2)/sqrt(size(A,1)-size(A,2))