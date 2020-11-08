function [r,J,c]=fun_All(a,t,y)

% obtain F(a)
a = reshape(a,length(a),1);
F = exp(t*a');

% compute c
c = linearLSQ(F,y);

% compute the residual, i.e. the variable projection of y
r = y-F*c;

% compute the Jacobian
[Q,R] = qr(F,0);
H = diag(t)*F;
J = -Q*(R'\(diag(H'*r)-H'*F*diag(c)))-H*diag(c);
% Also can use this one
% J = -F*((F'*F)\(diag(H'*rvp)-H'*F*diag(c)))-H*diag(c); 


