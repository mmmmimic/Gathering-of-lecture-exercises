function [r, J] = func_rJ_exe(x)
%% Input: 
%      :x (vector)
x1 = x(1);
x2 = x(2);
r = [x1;
    10.*x1./(x1+0.1)+2.*x2.^2];
J = [1, 0;
    1./(x1+0.1).^2, 4.*x2];

end