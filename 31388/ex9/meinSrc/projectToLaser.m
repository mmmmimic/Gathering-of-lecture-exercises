function [ projectedLine, lineCov ] = projectToLaser( worldLine,poseIn, covIn)
%[projectedLine, lineCov] = PROJECTTOLASER(worldLine,poseIn,covIn) 
%Project a word line to the laser scanner frame given the
%world line, the robot pose and robot pose covariance. Note that the laser
%scanner pose in the robot frame is read globally
%   worldLine: The line in world coordinates
%   poseIn: The robot pose
%   covIn: The robot pose covariance
%
%   projectedLine: The line parameters in the laser scanner frame
%   lineCov: The covariance of the line parameters

%% Constants
global lsrRelPose % The laser scanner pose in the robot frame is read globally


%% Calculation
%projectedLine = [0,0];
%% World Frame -> Robot Frame
% alpha_r = alpha_w-theta_w
% r_r =r_w - x_w*cos(alpha_w)-y_w*sin(alpha_w)
alpha_w = worldLine(1);
r_w = worldLine(2);
x_w = poseIn(1);
y_w = poseIn(2);
theta_w = poseIn(3);
alpha_r = alpha_w-theta_w;
r_r =r_w - x_w*cos(alpha_w)-y_w*sin(alpha_w);

%% Robot Frame -> Laser Frame
x_r = lsrRelPose(1);
y_r = lsrRelPose(2);
theta_r = lsrRelPose(3);
alpha_l = alpha_r-theta_r;
r_l =r_r - x_r*cos(alpha_r)-y_r*sin(alpha_r);
alpha_l = norm_ang(alpha_l);
projectedLine = [alpha_l;r_l];

%% Return real covariance
%lineCov = zeros(2,2);
lineCov = lineCov_(worldLine, poseIn, covIn);
end
function ang = norm_ang(ang)
if ang < -pi
    ang = ang+2*pi;
    ang = norm_ang(ang);
end
if ang > pi
    ang = ang-2*pi;
    ang = norm_ang(ang);
end
end