function [r, J]=fun_rJ_Rosen(x)
x1 = x(1);
x2 = x(2);
r = sqrt(2) * [10*(x2 - x1^2); 1 - x1];
J = [-20*sqrt(2)*x1, 10*sqrt(2);
    -sqrt(2), 0];
end