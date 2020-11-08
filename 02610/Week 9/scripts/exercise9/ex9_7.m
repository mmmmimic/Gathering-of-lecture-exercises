%% Change of variables
clear; close; clc;

%% Calculate the new Jacobian 
syms z1 z2;
rz = [z1; 10*z1/(z1+0.1)+2*z2];
J = jacobian(rz, [z1, z2])
det(J)
% The determinant of J is always 2,  J is nonsingular for all z.

%% Apply the Levenberg-Marquardt method
fun_rJ = @func_rJ_exe_new;
x0 = [3; 1];

[x,stat] = Levenberg_Marquardt_yq(fun_rJ, x0, 1e-16);

figure;
plot(stat.X);
xlabel('iteration');
ylabel('e_k');

figure;
plot(stat.F);
xlabel('iteration');
ylabel('1/2||x_k||_2^2');

tmp = [];
for i = 2:length(stat.X)
   tmp = [tmp, stat.X(i) / stat.X(i-1)^2]; 
end
figure;
plot(tmp);
xlabel('iteration');
ylabel('e(x_(k+1))/e(x_k)');

% Now I only need 2 iterations, and the convergence rate is quadratic. 