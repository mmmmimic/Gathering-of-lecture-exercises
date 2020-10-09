x0 = [1.2;1.2];
sl = [1e-3, 1e-2, 5e-2, 1e-1];% step length
figure,
for i = 1:4
    [x,stat] = steepestdescent(sl(i),'rosenbrock', x0);
    subplot(2,1,1),
    semilogy(0:stat.iter, stat.dX);
    hold on;
    title('e_k of fixed_length SD')
    legend('1e-3', '1e-2', '5e-2', '1e-1');
    subplot(2,1,2),
    semilogy(0:stat.iter, stat.dX);
    hold on;
    title('f(x_k) of fixed_length SD')
    legend('1e-3', '1e-2', '5e-2', '1e-1');
end

% In most cases, the values are approching 0. 
% It doesn't always coverage, such as the case when length = 0.05 and 0.1. 

x0 = [-1.2;1];
sl = [1e-3, 1e-2, 5e-2, 1e-1];% step length
figure,
for i = 1:4
    [x,stat] = steepestdescent(sl(i),'rosenbrock', x0);
    subplot(2,1,1),
    semilogy(0:stat.iter, stat.dX);
    hold on;
    title('e_k of fixed_length SD')
    legend('1e-3', '1e-2', '5e-2', '1e-1');
    subplot(2,1,2),
    semilogy(0:stat.iter, stat.dX);
    hold on;
    title('f(x_k) of fixed_length SD')
    legend('1e-3', '1e-2', '5e-2', '1e-1');
end

% It doesn't always coverage, such as the case when length = 0.05 and 0.1.
% I tried 4 different step lengths, 1e-3, 1e-2. 5e-2 and 1e-1. And I draw a
% conclusion that: 
% With larger step length, the optimization works faster, but with the
% higher probability of no coverage. 

% The closer point coverage faster than the futher point. 