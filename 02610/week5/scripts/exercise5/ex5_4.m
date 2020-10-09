clear all;close all;clc;
x1=-1:0.05:2;
x2=-1:0.05:2;
[X,Y]=meshgrid(x1,x2);
F=100*(Y-X.^2).^2+(1-X).^2;

figure,
v = [0:2:10, 10:10:100, 100:100:2500];
[c,h]=contour(X,Y,F,v,'linewidth',2);
hold on;
scatter(1,1,'r', 'linewidth', 2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),