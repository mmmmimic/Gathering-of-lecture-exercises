function [r, J] = func_rJ_exe_new(x)
%% Input: 
%      :x (vector)
x1 = x(1);
x2 = x(2);
r = [x1;
    10.*x1./(x1+0.1)+2.*x2];
J = [1, 0;
    1./(x1+0.1).^2, 2];

end