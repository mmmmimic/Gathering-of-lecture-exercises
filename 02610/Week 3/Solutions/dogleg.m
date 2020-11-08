function p = dogleg(df,d2f,delta)
% Inputs: 
% df:      the gradient (column vector)
% d2f:    the Hessian (matrix)
% delta: the trust-region radius
%
% Output:
% p:       the approximated minimizer to the subproblem through dogleg.


pu = - ((df'*df)/(df'*d2f*df))*df;
norm_pu = norm(pu);
pb = - d2f\df;
norm_pb = norm(pb);

if norm_pu >= delta
    p = (delta/norm_pu)*pu;
elseif norm_pb <= delta
    p = pb;
else
    tmp1 = pb-pu;
    tmp2 = 2*pu-pb;
    a = tmp1'*tmp1;
    b = 2*tmp1'*tmp2;
    c = tmp2'*tmp2-delta^2;
    
    chk = sqrt(b^2-4*a*c);
    tau1 = (-b+chk)/2/a;
    tau2 = (-b-chk)/2/a;
    
    if tau1 <= 2 && tau1 >= 1
        p = pu+(tau1-1)*tmp1;
    elseif tau2 <=2 && tau2 >= 1
        p = pu+(tau2-1)*tmp1;
    end
end