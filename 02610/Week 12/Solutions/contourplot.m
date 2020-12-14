function contourplot(t)

x=0:0.005:3;
y=(t-1.5):0.05:(t+1.5);
[X,Y]=meshgrid(x,y);
F=(X-1.5).^2+(Y-t).^4;

figure,
v=[0:0.02:0.1,0.2:0.1:1];
[c,h]=contour(X,Y,F,v,'linewidth',2);
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
colorbar

x1=0:0.05:1;
x2=-1:0.05:0;
yc1=1-x1;
yc2=x1-1;
yc3=x2+1;
yc4=-x2-1;
hold on
plot(x1, yc1,'b',x1,yc2,'b',x2,yc3,'b',x2,yc4','b','linewidth',2)
xL = xlim;
yL = ylim;
line([0 0], yL);  
line(xL, [0 0]);
axis equal
hold off