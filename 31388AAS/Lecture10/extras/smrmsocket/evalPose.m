function [ relPose ] = evalPose( absPose )
% function [ relPose ] = evalPose( absPose )
% Evaluate a given raw pose wrt the current reference pose. To read more
% about the reference pose see resetOdo function.
%
% absPose: the pose to be evaluated
%
% relPose: the pose relative to the reference pose


relPose=[0 0 0];

global x0 y0 th0
if isempty(x0) || isempty(y0) ||isempty(th0) 
    resetOdo(0,[0 0 0], [0 0 0]);
end

relPose(3) = mod(absPose(3)-th0+pi,2*pi)-pi;

xy=[cos(th0) sin(th0); -sin(th0) cos(th0)]*[absPose(1)-x0;absPose(2)-y0];
relPose(1:2)=xy';
