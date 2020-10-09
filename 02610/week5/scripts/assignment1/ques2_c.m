x1 =  -4:0.05:0;
x2 =  -3:0.05:3;
[X1,X2]=meshgrid(x1,x2);
F=(X2.^2+X1-1).^2+(X1+3).^2;

figure,
v = 0:20;
[c,h]=contour(X1,X2,F,v,'linewidth',2);
hold on;
scatter([-1, -3, -3],[0,-2,2],'r', 'linewidth', 2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),