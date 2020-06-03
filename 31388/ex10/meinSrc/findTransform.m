function transform = findTransform(odoPose, pose)
% transform = FINDTRANSFORM(odoPose,pose)
% Find the transformation from the world coordinates to the odometry
% coordinates given a pose in the odometry coordinates (odoPose) and the
% same point in the world coordinates (pose). The output (transform) is
% simply the origo of the odometry coordinates in the world coordinates
xy_o = odoPose(1:2);
theta_o = odoPose(3);
xy_w = pose(1:2);
theta_w = pose(3);
theta_T = norm_ang(theta_o-theta_w);
xy_T = xy_o-[cos(theta_T),-sin(theta_T);sin(theta_T),cos(theta_T)]*xy_w;
transform = [xy_T;theta_T];
%transform = [0;0;0];
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