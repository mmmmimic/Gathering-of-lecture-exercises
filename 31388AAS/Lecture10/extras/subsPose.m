function [ poseRel ] = subsPose( pose1, pose2 )
%ADDPOSE Summary of this function goes here
%   Detailed explanation goes here

x=pose1(1);
y=pose1(2);
th = pose1(3);

rot = [cos(th) sin(th); -sin(th) cos(th)];

poseRel=[rot*([pose2(1);pose2(2)]-[x; y]); pose2(3)-th];

end
