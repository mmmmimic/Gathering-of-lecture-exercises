function odoTargetPose = trans(transform,targetPose)
% odoTargetPose = trans(transform,targetPose)
% Transform a given point in world coordinates (targetPose) to odometry
% coordinates, using the origo of the odometry coordinates in world
% coordinates (transform).
xy_w = targetPose(1:2);
theta_w = targetPose(3);
xy_T = transform(1:2);
theta_T = transform(3);
xy_o = [cos(theta_T),-sin(theta_T);sin(theta_T),cos(theta_T)]*xy_w+xy_T;
theta_o = theta_w+theta_T;
odoTargetPose = [xy_o;theta_o];
%odoTargetPose = [0;0;0];
end