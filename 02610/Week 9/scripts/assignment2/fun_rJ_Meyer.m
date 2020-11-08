function [r, J]=fun_rJ_Meyer(x, y, t)
x1 = x(1); x2 = x(2); x3 = x(3);
r = y - x1 .* exp(x2./(t+x3));
J = [-exp(x2./(t+x3)), -(x1.*exp(x2./(t+x3)))./(t+x3), (x1.*x2.*exp(x2./(t+x3)))./(t+x3).^2];
end