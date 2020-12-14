x=0:0.05:5;
y=0:0.05:5;
[X,Y]=meshgrid(x,y);
F=-2*X-Y;

figure,
%v=[0:0.02:0.1,0.2:0.1:1];
[c,h]=contour(X,Y,F,30,'linewidth',2);
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
colorbar

%%
yc1=4-x;
hold on
fill([x(1),x,x(end)],[0,yc1,0],[0.3,0.3,0.3],'facealpha',0.5);
hold off
axis([0, 5, 0, 5])

%%

f=[-2;-1];
A=[-1 0;0 -1;1 1];
b=[0; 0; 4];
x=linprog(f,A,b);

hold on
plot(x(1),x(2),'b*','markersize',20)
hold off