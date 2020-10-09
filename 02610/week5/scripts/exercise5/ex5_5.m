clear all;close all;clc;
x1=-1:0.05:2;
x2=-1:0.05:2;
[X,Y]=meshgrid(x1,x2);
F=log(100*(Y-X.^2).^2+(1-X).^2+1e-24);

figure,
v = [-28:0.5:8];
[c,h]=contour(X,Y,F,v,'linewidth',2);
hold on;
scatter(1,1,'r', 'linewidth', 1);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14)

% Now the information is more clear
% The two figures look totally different, the log figure is more detailed