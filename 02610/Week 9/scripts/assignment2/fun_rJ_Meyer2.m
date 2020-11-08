function [r, J] = fun_rJ_Meyer2(z, u, y)
z1 = z(1); z2 = z(2); z3 = z(3); 
r = 1e-3.*y - z1.*exp(10.*z2./(u+z3)-13);
J = [-exp(10.*z2./(u+z3)-13),...
    -10.*z1.*exp(10.*z2./(u+z3)-13)./(u+z3),...
    10.*z1.*z2.*exp(10.*z2./(u+z3)-13)./(u+z3).^2];
end