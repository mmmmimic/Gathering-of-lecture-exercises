%%
clear; close all; clc;
%% Exercise 5.1
%// cubic splines
t1 = 0;
t2 = 1;
A = [1, t1, t1^2, t1^3;
    0, 1, 2*t1, 3*t1^2;
    1, t2, t2^2, t2^3;
    0, 1, 2*t2, 3*t2^3];
B = [0;0;1;0];
a = pinv(A)*B;
t = 0:0.001:1;
x = a(1)+a(2)*t+a(3)*t.^2+a(4)*t.^3;
v = a(2)+2*a(3)*t+3*a(4)*t.^2;
figure;
subplot(2,1,1);
plot(t, x);
xlabel('time/s');
ylabel('displacement/m');
title('Displacement');
subplot(2,1,2);
plot(t, v);
xlabel('time/s');
ylabel('velocity/(m/s)');
title('Velocity');

%% Exercise 5.2
% // quintic splines
t1 = 0;
t2 = 1;
A = [1, t1, t1^2, t1^3, t1^4, t1^5;
    0, 1, 2*t1, 3*t1^2, 4*t1^3, 5*t1^4;
    0, 0, 2, 6*t1, 12*t1^2, 20*t1^3;
    1, t2, t2^2, t2^3, t2^4, t2^5;
    0, 1, 2*t2, 3*t2^2, 4*t2^3, 5*t2^4;
    0, 0, 2, 6*t2, 12*t2^2, 20*t2^3];
B = [0;0;0;1;0;0];
a = pinv(A)*B;
t = 0:0.001:1;
x = a(1)+a(2)*t+a(3)*t.^2+a(4)*t.^3+a(5)*t.^4+a(6)*t.^5;
v = a(2)+2*a(3)*t+3*a(4)*t.^2+4*a(5)*t.^3+5*a(6)*t.^4;
acc = 2*a(3)+6*a(4)*t+12*a(5)*t.^2+20*a(6)*t.^3;
figure;
subplot(3,1,1);
plot(t, x);
xlabel('time/s');
ylabel('displacement/m');
title('Displacement');
subplot(3,1,2);
plot(t, v);
xlabel('time/s');
ylabel('velocity/(m/s)');
title('Velocity');
subplot(3,1,3);
plot(t, acc);
xlabel('time/s');
ylabel('acceleration/(m/s^2)');
title('Acceleration');
%% Exercise 5.3
% // 