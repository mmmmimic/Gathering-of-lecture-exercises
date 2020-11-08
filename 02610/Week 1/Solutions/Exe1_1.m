f = @(x)(x-log(x));
df = @(x) (1- 1./x);
d2f = @(x) 1./(x.^2);

x=0.01:0.01:2;
y=f(x);
dy= df(x);
d2y=d2f(x);

figure, 
subplot(1,3,1), plot(x,y,'b-',1,1,'ro'), title('f(x)'),grid on,
subplot(1,3,2), plot(x,dy), title('1st-order derivative'),grid on,
subplot(1,3,3), semilogy(x,d2y), title('2nd-order derivative'),grid on,


disp('The minimizer is x=1, which can be obtained by solving f''(x)=0.')
disp('The minimizer is unique, since d2f>0, i.e., f strictly convex.')