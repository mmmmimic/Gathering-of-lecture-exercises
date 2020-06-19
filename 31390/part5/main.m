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
% // Cox-de Boor recusion
% compute the two first order functions N(0,1,t) and N(1,1,t)
% show the lines
order = 1;
n1 = [];
t = [];
for i = 0:0.01:10
    n1 = [n1, N(0, order, i)];
    t = [t, i];
end
figure;
plot(t, n1);
hold on;
n2 = [];
t = [];
for i = 0:0.01:10
    n2 = [n2, N(1, order, i)];
    t = [t, i];
end
plot(t, n2);
xlabel('x');
ylabel('y');
% combine them linearly
% we need to calculate weight
A = [N(0,order,1), N(0,order,2);N(1,order,1),N(1,order,2)];
B = [2, 3];
W = B*pinv(A);
% the line we want
figure;
n3 = [];
t = [];
for i = 1:0.01:2
    n3 = [n3, W*[N(0,order,i);N(1,order,i)]];
    t = [t, i];
end
plot(t, n3);
xlabel('x');
ylabel('y');

%% Exercise 5.4
syms psi theta phi
Rx = [1 0 0;
    0 cos(phi) -sin(phi);
    0 sin(phi) cos(phi)];
Ry = [cos(theta) 0 sin(theta);
    0 1 0;
    -sin(theta) 0 cos(theta)];
Rz = [cos(psi) -sin(psi) 0;
    sin(psi) cos(psi) 0;
    0 0 1];
zB = simplify(Rz*Ry*Rx*[0;0;1]);
xC = simplify(Rz*[1;0;0]);
yB = simplify(Rz*Ry*Rx*[0;1;0]);
YB = simplify(cross(zB, xC));
transpose(xC)*YB
% it's 0, perpendicular proved
%% Exercise 5.5

%%
function n = N(i, p, t)
% since t_i = i
if p==0
 if i<=t && t<(i+1)
    n = 1;
    return;
else
    n = 0;
    return;
end   
end
n = (t-i)/p*N(i, p-1, t)+(i+p+1-t)/p*N(i+1, p-1, t);
end