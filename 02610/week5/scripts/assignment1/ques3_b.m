x1 =  -3:0.05:3;
x2 =  -3:0.05:3;
[X1,X2]=meshgrid(x1,x2);
F=X1.^4/4-X1.^2+2*X1+(X2-1).^2;

figure,
v = -5:34;
[c,h]=contour(X1,X2,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),