function [sigmazp] = lineCov_(zw, pose, poseCov)
%Output sigmazp is the covariance matrix of the predicted line features
%Inputs
%zw is the line features in world coordinates
%pose is the predicted robot pose in world coordinates
%poseCov is the covariance matrix of the predicted robot pose
alpha_w = zw(1);
global lsrRelPose % The laser scanner pose in the robot frame is read globally
x_l = lsrRelPose(1);
y_l = lsrRelPose(2);
theta = pose(3);
grad_h = [0,0,-1;-cos(alpha_w), -sin(alpha_w), -x_l*sin(alpha_w-theta)+y_l*cos(alpha_w-theta)];
sigmazp = grad_h*poseCov*grad_h';
end