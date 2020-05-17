function [ poseOut ] = addPose( poseIn, relPose )
%ADDPOSE Summary of this function goes here
%   Detailed explanation goes here

x=poseIn(1);
y=poseIn(2);
th = poseIn(3);

rot = [cos(th) -sin(th); sin(th) cos(th)];

poseOut=[(rot*[relPose(1);relPose(2)])+[x; y]; th+relPose(3)];

end
