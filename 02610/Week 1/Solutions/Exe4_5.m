x1=0:0.05:5;
x2=x1;
[X,Y]=meshgrid(x1,x2);
F=X.^2+Y.^2;

figure,
v=[-20:20];
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),

disp('The feasible set is a convext set.')

%%

f = @(x)(x.*log(x));
df = @(x)(log(x)+1);
d2f = @(x)(1./x);

x=0.01:0.01:5;
y=f(x);
dy= df(x);
d2y=d2f(x);

figure, 
subplot(1,3,1), plot(x,y,'b-',1,1), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y), title('2nd-order derivative'),grid on,

disp('The objective function is a convex function in the feasible set.')