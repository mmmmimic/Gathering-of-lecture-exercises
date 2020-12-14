x=-1:0.005:1;
y=x;
[X,Y]=meshgrid(x,y);
F=X.*Y;

figure,
%v=[0:0.02:0.1,0.2:0.1:1];
[c,h]=contour(X,Y,F,30,'linewidth',2);
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
colorbar


hold on, ezplot('x^2+y^2=1'), hold off,
axis([-1, 1, -1, 1])