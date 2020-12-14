clc, clear all, close all

dim = [5,10,20,30];

for ii=1:length(dim)
    n = dim(ii);
    A=gallery('poisson',n);
    condest(A)
    b = ones(n^2,1);
    x0 = zeros(n^2,1);

    [x,stat]=cgm(A,b,x0);

    figure,
    semilogy(stat.resd), title('Residual'),
    
end


%% Steepest descent
for ii=1:length(dim)
    n = dim(ii);
    A=gallery('poisson',n);
    b = ones(n^2,1);
    x0 = zeros(n^2,1);
         
    [x_SD,stat_SD] = steepestdescent_line(@funExe2,x0,A,b);
    
    figure,
    semilogy(stat_SD.normdF), title('Residual'),
end