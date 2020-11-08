clc, clear all, close all,

load data_exe4

n=2;
A2=[t, ones(size(t))];
c2=linearLSQ(A2,y);
r2=y-A2*c2;
std2=norm(r2)/sqrt(size(A2,1)-size(A2,2));

n=3;
A3=[t.^2, t, ones(size(t))];
c3=linearLSQ(A3,y);
r3=y-A3*c3;
std3=norm(r3)/sqrt(size(A3,1)-size(A3,2));

n=4;
A4=[t.^3, t.^2, t, ones(size(t))];
c4=linearLSQ(A4,y);
r4=y-A4*c4;
std4=norm(r4)/sqrt(size(A4,1)-size(A4,2));

n=5;
A5=[t.^4, t.^3, t.^2, t, ones(size(t))];
c5=linearLSQ(A5,y);
r5=y-A5*c5;
std5=norm(r5)/sqrt(size(A5,1)-size(A5,2));

n=6;
A6=[t.^5,t.^4, t.^3, t.^2, t, ones(size(t))];
c6=linearLSQ(A6,y);
r6=y-A6*c6;
std6=norm(r6)/sqrt(size(A6,1)-size(A6,2));

figure,
plot(t,y,'ro',t,A2*c2,'b',t,A3*c3,'g',t,A4*c4,'k',t,A5*c5,'c',t,A6*c6,'m'),
legend('data','n=2','n=3','n=4','n=5','n=6')

figure,
plot([2,3,4,5,6],[std2,std3,std4,std5,std6],'r*')
disp('From n=5, the standard deviation estimate levels off, i.e., the optimal n is 5.')
