function Exe6()

x=-5:0.005:5;
y=x;
[X,Y]=meshgrid(x,y);
F=(X.^2+Y-11).^2+(X+Y.^2-7).^2;

v=[0:2:10 10:10:100 100:20:200];
figure,
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar

yc1=(x+2).^2;
yc2=(4*x)/10;
hold on
fill(x,yc1,[0.3,0.3,0.3],'facealpha',0.5);
fill([x x(end) x(1)],[yc2 -5 -5],[0.3,0.3,0.3],'facealpha',0.5)
axis([-5 5 -5 5])

objf=@(x)(x(1)^2+x(2)-11)^2+(x(1)+x(2)^2-7)^2;
x1=fmincon(objf,[0;0],[],[],[],[],[],[],@conf)
x2=fmincon(objf,[-3;-1],[],[],[],[],[],[],@conf)
x3=fmincon(objf,[-3;3],[],[],[],[],[],[],@conf)
x4=fmincon(objf,[0;3],[],[],[],[],[],[],@conf)

minx=[x1,x2,x3,x4];

plot(minx(1,:),minx(2,:),'b.','markersize',20);
hold off

function [c,ceq]=conf(x)
c=[x(2)-(x(1)+2)^2; 4*x(1)-10*x(2)];
ceq=[];