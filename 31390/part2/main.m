clear; close all; clc;
%% Exercise 2.1
% the rotation matrix
% Row-Pitch-Yaw angular representation
syms phi theta psi
% THETA = [phi, theta, psi]
R = simplify(rotz(psi)*roty(theta)*rotx(phi));
%%
% the relation between the angular velocity and the rotational velocity of
% the body-fixed frame
% w = [1, 0, -sin(theta);
% 0, cos(phi), cos(theta)*sin(phi);
% 0, -sin(phi), cos(theta)*cos(phi)]*dot{theta}
%%
%// the linear dynamic equation
% m*ddot{p} = m*g+R*[0;0;SUM{k*w^2}]-D*dot{p}
%// the angular dynamic equation
% drag torque along z axis: tau_z = b*(w1^2-w2^2+w3^2-w4^2)
% lift torque along x axis: tau_x = k*(w1^2-w3^2)*L
% lift torque along y axis: tau_y = k*(w2^2-w4^2)*L
% dot{w} = [tau_x*Ixx^-1;
% tau_y*Iyy^-1;
% tau_z*Izz^-1]-[(Iyy-Izz)/Ixx*wy*wz;
% (Izz-Ixx)/Iyy*wx*wz;
% (Ixx-Iyy)/Izz*wx*wy];
%%
%// inputs
% static: 11.0736x4
OMEGA = [0;10000;0;10000];
%// plot param
lim = 1e-2;
%// simulation param
running_time = 1e-2;
samp_time = 1e-3;
% initialization
m = 0.5;
L = 0.225;
k = 0.01;
b = 0.001;
D = diag([0.01,0.01,0.01]);
I = diag([3e-6, 3e-6, 1e-5]);
Euler = [0;0;0];
p = [0;0;0];
ORIENTATION = [Euler];
POSITION = [p];
dEuler = [0;0;0];
dp = [0;0;0];
g = [0;0;-9.81];
w = [1, 0, -sin(Euler(2));
    0, cos(Euler(1)), cos(Euler(2))*sin(Euler(1));
    0, -sin(Euler(1)), cos(Euler(2))*cos(Euler(1))]*dEuler;


iter = round(running_time/samp_time);
for i = 1:iter
    R = rotz(Euler(3))*roty(Euler(2))*rotx(Euler(1));
    ddp = (m*g+R*[0;0;k*sum(OMEGA.^2)]-D*dp)/m;
    % torque
    tau = [k*(OMEGA(1)^2-OMEGA(3)^2)*L;
        k*(OMEGA(2)^2-OMEGA(4)^2)*L;
        b*(OMEGA(1)^2-OMEGA(2)^2+OMEGA(3)^2-OMEGA(4)^2)];
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
    Euler = Euler+dEuler*samp_time;
    Euler(1) = wrapToPi(Euler(1));
    Euler(2) = wrapToPi(Euler(2));
    Euler(3) = wrapToPi(Euler(3));
    ORIENTATION = [ORIENTATION,Euler];
end

% show the result
figure;
subplot(2,1,1);
plot3(ORIENTATION(1,:), ORIENTATION(2,:), ORIENTATION(3,:));
xlabel('w_x');
ylabel('w_y');
zlabel('w_z');
xlim([0,2*pi]);
ylim([0,2*pi]);
zlim([0,2*pi]);
title("Orientation");
grid on;
subplot(2,1,2);
plot3(POSITION(1,:), POSITION(2,:), POSITION(3,:));
xlabel('x');
ylabel('y');
zlabel('z');
xlim([-lim,lim]);
ylim([-lim,lim]);
zlim([-lim,lim]);
grid on;
title("Position");
figure;
subplot(2,1,1);
plot([0:iter]*samp_time,ORIENTATION(1,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(2,:));
hold on;
plot([0:iter]*samp_time,ORIENTATION(3,:));
xlabel('time/s');
ylabel('radian');
legend('phi', 'theta', 'psi');
ylim([-pi,pi]);
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
%% Exercise 2.3
% INPUT
% U1 = k*sum(OMEGA.^2);
% U2 = k*(OMEGA(1)^2-OMEGA(3)^2)*L;
% U3 = k*(OMEGA(2)^2-OMEGA(4)^2)*L;
% U4 = b*(OMEGA(1)^2-OMEGA(2)^2+OMEGA(3)^2-OMEGA(4)^2);
% ddx = (-U1*(sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta))-d*dx)/m;
% ddy = (-U1*(cos(phi)*sin(psi)*sin(theta) - cos(psi)*sin(phi))-d*dy)/m;
% ddz = (m*g-U1*(cos(phi)*cos(theta))-d*dz)/m;
% //
% wx = dphi - sin(theta)*dpsi;
% wy = cos(phi)*dtheta+cos(theta)*sin(phi)*dpsi;
% wz = -sin(phi)*dtheta+cos(theta)*cos(phi)*dpsi;
% Assume dw = 0
% dwx = U2/Ixx-(Iyy-Izz)/Ixx*wy*wz;
% dwy = U3/Iyy-(Izz-Ixx)/Iyy*wx*wz;
% dwz = U4/Izz-(Iyy-Izz)/Izz*wx*wy;
% //
% dphi = wx+sin(phi)*tan(theta)*wy+cos(phi)*tan(theta)*wz;
% dtheta = cos(phi)*wy-sin(phi)*wz;
% dpsi = sin(phi)/cos(theta)*wy+cos(phi)/cos(theta)*wz;
%% Linearization
% Taylor Expansion
% sin(x) = sin(x_0)+cos(x_0)*dx
% cos(x) = cos(x_0)-sin(x_0)*dx
% We start at the hover pose: ORIENTATION, POSITION are all 0
% that is, x_0 = 0
% Thus we have
% sin(x) = dx
% cos(x) = 1
% x can be any phi/theta/psi
% Therefore
% ddx = (-U1*(dphi*dpsi + dtheta)-d*dx)/m;
% ddy = (-U1*(dpsi*dtheta - dphi)-d*dy)/m;
% ddz = (m*g-U1-d*dz)/m;
% dphi = wx+dphi*dtheta*wy+dtheta*wz;
% dtheta = wy-dphi*wz;
% dpsi = dphi*wy+wz;
% wx = dphi - dtheta*dpsi;
% wy = dtheta + dphi*dpsi;
% wz = -dphi*dtheta + dpsi;
% dwx = U2/Ixx-(Iyy-Izz)/Ixx*wy*wz;
% dwy = U3/Iyy-(Izz-Ixx)/Iyy*wx*wz;
% dwz = U4/Izz-(Iyy-Izz)/Izz*wx*wy;
%% SIMULATION
%%
%// inputs
% static: 11.0736x4
OMEGA = [0;10000;0;10000];
%// plot param
lim = 1e-2;
%// simulation param
running_time = 1e-2;
samp_time = 1e-3;
% initialization
m = 0.5;
L = 0.225;
k = 0.01;
b = 0.001;
D = diag([0.01,0.01,0.01]);
I = diag([3e-6, 3e-6, 1e-5]);
Euler = [0;0;0];
p = [0;0;0];
ORIENTATION = [Euler];
POSITION = [p];
dEuler = [0;0;0];
dp = [0;0;0];
g = [0;0;-9.81];
w = [1, 0, -dEuler(2);
    0, 1, dEuler(1);
    0, -dEuler(1), 1]*dEuler;

iter = round(running_time/samp_time);
for i = 1:iter
    R = [ 1, dEuler(1)*dEuler(2) - dEuler(3), dEuler(1)*dEuler(3) + dEuler(2);
        dEuler(3), 1 + dEuler(1)*dEuler(3)*dEuler(2), dEuler(3)*dEuler(2) - dEuler(1);
        -dEuler(2),   dEuler(1), 1];
    ddp = (m*g+R*[0;0;k*sum(OMEGA.^2)]-D*dp)/m;
    % torque
    tau = [k*(OMEGA(1)^2-OMEGA(3)^2)*L;
        k*(OMEGA(2)^2-OMEGA(4)^2)*L;
        b*(OMEGA(1)^2-OMEGA(2)^2+OMEGA(3)^2-OMEGA(4)^2)];
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
    Euler = Euler+dEuler*samp_time;
    Euler(1) = wrapToPi(Euler(1));
    Euler(2) = wrapToPi(Euler(2));
    Euler(3) = wrapToPi(Euler(3));
    ORIENTATION = [ORIENTATION,Euler];
end

% show the result
figure;
subplot(2,1,1);
plot3(ORIENTATION(1,:), ORIENTATION(2,:), ORIENTATION(3,:));
xlabel('w_x');
ylabel('w_y');
zlabel('w_z');
xlim([0,2*pi]);
ylim([0,2*pi]);
zlim([0,2*pi]);
title("Orientation");
grid on;
subplot(2,1,2);
plot3(POSITION(1,:), POSITION(2,:), POSITION(3,:));
xlabel('x');
ylabel('y');
zlabel('z');
xlim([-lim,lim]);
ylim([-lim,lim]);
zlim([-lim,lim]);
grid on;
title("Position");
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
