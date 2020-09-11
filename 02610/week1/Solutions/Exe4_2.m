clear all, close all,

%% f1
f1 = @(x)(x.^2+x+1);
df1 = @(x) (2*x+1);
d2f1 = @(x) 2*ones(length(x));

x=-2:0.01:2;
y1=f1(x);
dy1= df1(x);
d2y1=d2f1(x);

figure, 
subplot(1,3,1), plot(x,y1,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy1), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y1), title('2nd-order derivative'),grid on,


%% f2

f2 = @(x)(-x.^2+x+1);
df2 = @(x) (-2*x+1);
d2f2 = @(x) -2*ones(length(x));

x=-2:0.01:2;
y2=f2(x);
dy2= df2(x);
d2y2=d2f2(x);

figure, 
subplot(1,3,1), plot(x,y2,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy2), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y2), title('2nd-order derivative'),grid on,

%% f3

f3 = @(x)(x.^3-5*x.^2+x+1);
df3 = @(x) (3*x.^2-10*x+1);
d2f3 = @(x) 6*x-10;

x=-4:0.01:4;
y3=f3(x);
dy3= df3(x);
d2y3=d2f3(x);

figure, 
subplot(1,3,1), plot(x,y3,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy3), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y3), title('2nd-order derivative'),grid on,


%% f4

f4 = @(x)(x.^4+x.^3-10*x.^2-x+1);
df4 = @(x) (4*x.^3+3*x.^2-10*x-1);
d2f4 = @(x) 12*x.^2+6*x-10;

x=-4:0.01:4;
y4=f4(x);
dy4= df4(x);
d2y4=d2f4(x);

figure, 
subplot(1,3,1), plot(x,y4,'b-'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy4), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y4), title('2nd-order derivative'),grid on,


disp('f1 is convex.')
disp('At local minimizer, the value of the 2nd derivative is positive.')
dips('At local maximizer, the value of the 2nd derivative is negative.')


