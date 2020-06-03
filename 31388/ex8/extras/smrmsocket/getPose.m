function [ pose,absPose ] = getPose( socket,abs )
% [pose, absPose] = getPose(socket, abs)
% Convenience function to receive odometry over a socket connection to an
% smrdemo server. Note that the returned value does NOT equal to the value
% coming from the server unless absolute value is asked by inputing the
% string 'abs' as the second input. It instead equals to the position with
% respect to the reference pose. In order to understand what the reference 
% pose is and to change it see resetOdo function. The reference pose
% provides convenience in measurements and driving
%
% socket: the handle object for the socket connection
% abs: an optional string, if given as the string 'abs' the returned value
%   is whatever the server sends, if omitted the pose is wrt the reference
%   pose
%
% pose: The resulting pose
% absPose: The pose coming from the server can be captured from this output
% regardless of the value of the abs input

global x0 y0 th0

if isempty(x0) || isempty(y0) ||isempty(th0) 
    resetOdo(socket,[0 0 0], [0 0 0]);
end

pose=[];
receiveMessage(socket,0.02);
mssendraw(socket,uint8(['eval $odox' 10]));
mssendraw(socket,uint8(['eval $odoy' 10]));
mssendraw(socket,uint8(['eval $odoth' 10]));
while(length(pose)<3)
    response=char(receiveMessage(socket,0.3,0.3));
    pose=[pose sscanf(response,'%f')']; %#ok<AGROW>
end

absPose=pose;

if nargin==1 || strcmp(abs,'abs')==0
    pose=evalPose(pose);
end
