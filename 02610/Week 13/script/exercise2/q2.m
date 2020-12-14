clear; close all; clc;

b = ones(500,1); 
x_0 = zeros(500,1);
taus = [0.01, 0.05, 0.09, 0.3];
taus = [0.3]
figure;
for i = 1:4
    tau = taus(i);
    A = genA(tau);
    [x, stat]=cgm(A, b, x_0);
    plot(0:stat.iter, log10(stat.resd));
    hold on;
end
hold off;
xlabel('iteration')
ylabel('log_{10}({||Ax_k-b||}_2)')
legend('tau = 0.01','tau = 0.05','tau = 0.09', 'tau = 0.3')
title('CG method Convergence')