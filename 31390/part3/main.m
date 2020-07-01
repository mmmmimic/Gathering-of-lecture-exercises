clear; close all; clc;
%% Exercise 3.1
%%
%// inputs of the system
% // desire values
%Euler_desire = [10/180*pi;0;0];
%Euler_desire = [0;10/180*pi;0];
%Euler_desire = [0;0;10/180*pi];
Euler_desire = [0;0;0];
z_desire = 1;
% // since we are only able to know the orientation of the UAV in body
% frame
Euler = [0;0;0];
dEuler = [0;0;0];
p = [0;0;0];
dp = [0;0;0];
ORIENTATION = [Euler];
POSITION = [p];
OMEGA = [0;0;0;0];

w = [1, 0, -sin(Euler(2));
    0, cos(Euler(1)), cos(Euler(2))*sin(Euler(1));
    0, -sin(Euler(1)), cos(Euler(2))*cos(Euler(1))]*dEuler;
%// plot param
lim = 1e-2;
% //Attitude PID cofficients
k_p = 0.7;
k_i = 0;
k_d = 1;

% // Altitude PID cofficients
k_pa = 0.9;
k_ia = 0;
k_da = 1.1;

% //initialization
m = 0.5;
L = 0.225;
k = 0.01;
b = 0.001;
D = diag([0.01,0.01,0.01]);
I = diag([3e-6, 3e-6, 1e-5]);
g = [0;0;-9.81];

%// simulation param
running_time = 10; %run the model for 0.5 second
samp_time = 1e-2;

% iteration
iter = round(running_time/samp_time);
% // history
dE = zeros(3,iter);

for i = 1:iter
    % control part
    ie = [trapz(samp_time, dE(1,:));
        trapz(samp_time, dE(2,:));
        trapz(samp_time, dE(3,:))];
    error_o = ie-Euler_desire;
    error_z = p(3)-z_desire;
    u_o = k_d*dEuler+k_p*error_o;
    u_z = -(k_da*(dp(3))+k_pa*error_z);
    hover_vec = (-m*g(3)+u_z+D(3,3)*dp(3))/(k*cos(Euler(2))*cos(Euler(1)));
    OMEGA(1) = hover_vec/4-(2*b*u_o(1)*I(1,1)+k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(2) = hover_vec/4-(2*b*u_o(2)*I(2,2)-k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(3) = hover_vec/4-(-2*b*u_o(1)*I(1,1)+k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(4) = hover_vec/4-(-2*b*u_o(2)*I(2,2)-k*L*I(3,3)*u_o(3))/(4*b*k*L);
    
    R = rotz(Euler(3))*roty(Euler(2))*rotx(Euler(1));
    ddp = (m*g+R*[0;0;k*sum(OMEGA)]-D*dp)/m;
    % torque
    tau = [k*(OMEGA(1)-OMEGA(3))*L;
        k*(OMEGA(2)-OMEGA(4))*L;
        b*(OMEGA(1)-OMEGA(2)+OMEGA(3)-OMEGA(4))];
    
    dw = pinv(I)*(tau-cross(w,I*w));
    % motion update
    %// position
    p = p+dp*samp_time+0.5*ddp*samp_time^2;
    dp = dp+ddp*samp_time;
    POSITION = [POSITION,p];
    
    %// orientation
    w = w+dw*samp_time;
    dEuler = [1, sin(Euler(1))*tan(Euler(2)), cos(Euler(1))*tan(Euler(2));
        0, cos(Euler(1)), -sin(Euler(1));
        0,sin(Euler(1))/cos(Euler(2)), cos(Euler(1))/cos(Euler(2))]*w;
    dE(:,i) = dEuler;
    Euler = Euler+dEuler*samp_time;
    Euler(1) = wrapToPi(Euler(1));
    Euler(2) = wrapToPi(Euler(2));
    Euler(3) = wrapToPi(Euler(3));
    ORIENTATION = [ORIENTATION,Euler];
end
% show the result
% figure;
% subplot(2,1,1);
% plot3(ORIENTATION(1,:), ORIENTATION(2,:), ORIENTATION(3,:));
% xlabel('w_x');
% ylabel('w_y');
% zlabel('w_z');
% xlim([0,2*pi]);
% ylim([0,2*pi]);
% zlim([0,2*pi]);
% title("Orientation");
% grid on;
% subplot(2,1,2);
% plot3(POSITION(1,:), POSITION(2,:), POSITION(3,:));
% xlabel('x');
% ylabel('y');
% zlabel('z');
% xlim([-lim,lim]);
% ylim([-lim,lim]);
% zlim([-lim,lim]);
% grid on;
% title("Position");

figure;
subplot(2,1,1);
plot([0:iter]*samp_time,ORIENTATION(1,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(2,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(3,:));
xlabel('time/s');
ylabel('radian');
ylim([-pi,pi]);
legend('phi', 'theta', 'psi');
title("Orientation");
subplot(2,1,2);
plot([0:iter]*samp_time,POSITION(1,:));
hold on;
plot([0:iter]*samp_time,POSITION(2,:));
hold on;
plot([0:iter]*samp_time,POSITION(3,:));
xlabel('time/s');
ylabel('meter');
legend('x', 'y', 'z');
title("Position");
%% Exercise 3.2
%%
%// inputs of the system
% // desire values
%Euler_desire = [10/180*pi;0;0];
%Euler_desire = [0;10/180*pi;0];
%Euler_desire = [0;0;10/180*pi];
Euler_desire = [0;0;0];
z_desire = 1;
% // since we are only able to know the orientation of the UAV in body
% frame
Euler = [0;0;0];
dEuler = [0;0;0];
p = [0;0;0];
dp = [0;0;0];
ORIENTATION = [Euler];
POSITION = [p];
OMEGA = [0;0;0;0];
%// plot param
lim = 1e-2;
% //Attitude PID cofficients
k_p = 0.7;
k_i = 0;
k_d = 1;

% // Altitude PID cofficients
k_pa = 0.9;
k_ia = 0;
k_da = 1.1;

% //initialization
m = 0.5;
L = 0.225;
k = 0.01;
b = 0.001;
D = diag([0.01,0.01,0.01]);
I = diag([3e-6, 3e-6, 1e-5]);
g = [0;0;-9.81];

%// simulation param
running_time = 10; %run the model for 0.5 second
samp_time = 1e-2;

w = [1, 0, -dEuler(2);
    0, 1, dEuler(1);
    0, -dEuler(1), 1]*dEuler;

iter = round(running_time/samp_time);
dE = zeros(3,iter);

for i = 1:iter
    % control part
    ie = [trapz(samp_time, dE(1,:));
        trapz(samp_time, dE(2,:));
        trapz(samp_time, dE(3,:))];
    error_o = ie-Euler_desire;
    error_z = p(3)-z_desire;
    u_o = k_d*dEuler+k_p*error_o;
    u_z = -(k_da*(dp(3))+k_pa*error_z);
    hover_vec = (-m*g(3)+u_z+D(3,3)*dp(3))/(k);
    OMEGA(1) = hover_vec/4-(2*b*u_o(1)*I(1,1)+k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(2) = hover_vec/4-(2*b*u_o(2)*I(2,2)-k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(3) = hover_vec/4-(-2*b*u_o(1)*I(1,1)+k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(4) = hover_vec/4-(-2*b*u_o(2)*I(2,2)-k*L*I(3,3)*u_o(3))/(4*b*k*L);
    
    R = [ 1, dEuler(1)*dEuler(2) - dEuler(3), dEuler(1)*dEuler(3) + dEuler(2);
        dEuler(3), 1 + dEuler(1)*dEuler(3)*dEuler(2), dEuler(3)*dEuler(2) - dEuler(1);
        -dEuler(2),   dEuler(1), 1];
    ddp = (m*g+R*[0;0;k*sum(OMEGA)]-D*dp)/m;
    % torque
    tau = [k*(OMEGA(1)-OMEGA(3))*L;
        k*(OMEGA(2)-OMEGA(4))*L;
        b*(OMEGA(1)-OMEGA(2)+OMEGA(3)-OMEGA(4))];
    dw = pinv(I)*(tau-cross(w,I*w));
    % motion update
    %// position
    p = p+dp*samp_time+0.5*ddp*samp_time^2;
    dp = dp+ddp*samp_time;
    POSITION = [POSITION,p];
    
    %// orientation
    w = w+dw*samp_time;
    dEuler = [1,dEuler(1)*dEuler(2), dEuler(2);
        0, 1, -dEuler(1);
        0,dEuler(1), 1]*w;
    dE(:,i) = dEuler;
    Euler = Euler+dEuler*samp_time;
    Euler(1) = wrapToPi(Euler(1));
    Euler(2) = wrapToPi(Euler(2));
    Euler(3) = wrapToPi(Euler(3));
    ORIENTATION = [ORIENTATION,Euler];
end
% show the result
% figure;
% subplot(2,1,1);
% plot3(ORIENTATION(1,:), ORIENTATION(2,:), ORIENTATION(3,:));
% xlabel('w_x');
% ylabel('w_y');
% zlabel('w_z');
% xlim([0,2*pi]);
% ylim([0,2*pi]);
% zlim([0,2*pi]);
% title("Orientation");
% grid on;
% subplot(2,1,2);
% plot3(POSITION(1,:), POSITION(2,:), POSITION(3,:));
% xlabel('x');
% ylabel('y');
% zlabel('z');
% xlim([-lim,lim]);
% ylim([-lim,lim]);
% zlim([-lim,lim]);
% grid on;
% title("Position");

figure;
subplot(2,1,1);
plot([0:iter]*samp_time,ORIENTATION(1,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(2,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(3,:));
xlabel('time/s');
ylabel('radian');
ylim([-pi,pi]);
title("Orientation");
legend('phi', 'theta', 'psi');
subplot(2,1,2);
plot([0:iter]*samp_time,POSITION(1,:));
hold on;
plot([0:iter]*samp_time,POSITION(2,:));
hold on;
plot([0:iter]*samp_time,POSITION(3,:));
xlabel('time/s');
ylabel('meter');
title("Position");
legend('x', 'y', 'z');

%% Exercise 3.3
%%
%// inputs of the system
% // desire values
%Euler_desire = [10/180*pi;0;0];
%Euler_desire = [0;10/180*pi;0];
%Euler_desire = [0;0;10/180*pi];
Euler_desire = [0;0;0];
p_desire = [1;1;1];
% // since we are only able to know the orientation of the UAV in body
% frame
Euler = [0;0;0];
dEuler = [0;0;0];
p = [0;0;0];
dp = [0;0;0];
ORIENTATION = [Euler];
POSITION = [p];
OMEGA = [0;0;0;0];

w = [1, 0, -sin(Euler(2));
    0, cos(Euler(1)), cos(Euler(2))*sin(Euler(1));
    0, -sin(Euler(1)), cos(Euler(2))*cos(Euler(1))]*dEuler;
%// plot param
lim = 1e-2;
% //Attitude PID cofficients
k_p = 0.7;
k_i = 0;
k_d = 1;

% // Altitude PID cofficients
k_pa = 0.9;
k_ia = 0;
k_da = 1.1;

% // x PID cofficients
% k_px = 0.01;
% k_ix = 0;
% k_dx = 0.05;
k_px = 0.008;
k_ix = 0;
k_dx = 0.04;

% //initialization
m = 0.5;
L = 0.225;
k = 0.01;
b = 0.001;
D = diag([0.01,0.01,0.01]);
I = diag([3e-6, 3e-6, 1e-5]);
g = [0;0;-9.81];

%// simulation param
running_time = 30; %run the model for 0.5 second
samp_time = 1e-2;

% iteration
iter = round(running_time/samp_time);
% // history
dE = zeros(3,iter);

for i = 1:iter
    R = rotz(Euler(3))*roty(Euler(2))*rotx(Euler(1));
    % control part
    ie = [trapz(samp_time, dE(1,:));
        trapz(samp_time, dE(2,:));
        trapz(samp_time, dE(3,:))];
    error_x = p(1) - p_desire(1);
    error_y = p(2) - p_desire(2);
    u_x = k_dx*(dp(1))+k_px*error_x;
    u_y = k_dx*(dp(2))+k_px*error_y;
    xy_desire = [0,1,0;-1,0,0]*R*[u_x;u_y;0];
    Euler_desire(1) = xy_desire(1);
    Euler_desire(2) = xy_desire(2); 
    error_o = ie - Euler_desire;
    error_z = p(3) - p_desire(3);
    u_o = k_d*dEuler+k_p*error_o;
    u_z = -(k_da*(dp(3))+k_pa*error_z);
    hover_vec = (-m*g(3)+u_z+D(3,3)*dp(3))/(k*cos(Euler(2))*cos(Euler(1)));
    OMEGA(1) = hover_vec/4-(2*b*u_o(1)*I(1,1)+k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(2) = hover_vec/4-(2*b*u_o(2)*I(2,2)-k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(3) = hover_vec/4-(-2*b*u_o(1)*I(1,1)+k*L*I(3,3)*u_o(3))/(4*b*k*L);
    OMEGA(4) = hover_vec/4-(-2*b*u_o(2)*I(2,2)-k*L*I(3,3)*u_o(3))/(4*b*k*L);
     
    ddp = (m*g+R*[0;0;k*sum(OMEGA)]-D*dp)/m;
    % torque
    tau = [k*(OMEGA(1)-OMEGA(3))*L;
        k*(OMEGA(2)-OMEGA(4))*L;
        b*(OMEGA(1)-OMEGA(2)+OMEGA(3)-OMEGA(4))];
    
    dw = pinv(I)*(tau-cross(w,I*w));
    % motion update
    %// position
    p = p+dp*samp_time+0.5*ddp*samp_time^2;
    dp = dp+ddp*samp_time;
    POSITION = [POSITION,p];
    
    %// orientation
    w = w+dw*samp_time;
    dEuler = [1, sin(Euler(1))*tan(Euler(2)), cos(Euler(1))*tan(Euler(2));
        0, cos(Euler(1)), -sin(Euler(1));
        0,sin(Euler(1))/cos(Euler(2)), cos(Euler(1))/cos(Euler(2))]*w;
    dE(:,i) = dEuler;
    Euler = Euler+dEuler*samp_time;
    Euler(1) = wrapToPi(Euler(1));
    Euler(2) = wrapToPi(Euler(2));
    Euler(3) = wrapToPi(Euler(3));
    ORIENTATION = [ORIENTATION,Euler];
end
% % show the result
% figure;
% subplot(2,1,1);
% plot3(ORIENTATION(1,:), ORIENTATION(2,:), ORIENTATION(3,:));
% xlabel('w_x');
% ylabel('w_y');
% zlabel('w_z');
% xlim([0,2*pi]);
% ylim([0,2*pi]);
% zlim([0,2*pi]);
% title("Orientation");
% grid on;
% subplot(2,1,2);
% plot3(POSITION(1,:), POSITION(2,:), POSITION(3,:));
% xlabel('x');
% ylabel('y');
% zlabel('z');
% xlim([-lim,lim]);
% ylim([-lim,lim]);
% zlim([-lim,lim]);
% grid on;
% title("Position");

figure;
subplot(2,1,1);
plot([0:iter]*samp_time,ORIENTATION(1,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(2,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(3,:));
xlabel('time/s');
ylabel('radian');
ylim([-pi,pi]);
title("Orientation");
legend('phi', 'theta', 'psi');
subplot(2,1,2);
plot([0:iter]*samp_time,POSITION(1,:));
hold on;
plot([0:iter]*samp_time,POSITION(2,:));
hold on;
plot([0:iter]*samp_time,POSITION(3,:));
xlabel('time/s');
ylabel('meter');
title("Position");
legend('x', 'y', 'z');