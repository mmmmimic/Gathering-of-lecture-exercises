% Exercise 2.2 Trigonometric fit

clear, clc

x = [11 25 26 31 33 36 47 58 75 79]';
y = [160 140 138 130 125 120 95 72 27 17]';
m = length(x);

omega = 2*pi/365;
F = [ ones(m,1) sin(omega*x) cos(omega*x) ];

c=linearLSQ(F,y);
%%

xx = (0:80)';
FF = [ ones(size(xx)) sin(omega*xx) cos(omega*xx) ];
yy = FF*c;

figure,subplot(3,2,1)
plot(xx,yy,'-b',x,y,'or','linewidth',2)
set(gca,'fontsize',15)
axis([0 80 0 200])

xx = (0:200)';
FF = [ ones(size(xx)) sin(omega*xx) cos(omega*xx) ];
yy = FF*c;

[minyy,I] = min(yy);

subplot(3,2,2)
plot(xx,yy,'-b',x,y,'or','linewidth',2);
hold on
plot(xx(I),yy(I),'kd','linewidth',2)
hold off
set(gca,'fontsize',15)


text(90,100,['Min. at \itt\rm = ',num2str(xx(I))],'fontsize',15)