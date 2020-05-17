function resetOdo(socket,pose0,curpose)
% function resetOdo(socket,pose0,curpose)
% The function used to change the reference frame. The reference pose is
% used simply for flexibility in odometric measurements. Three global
% variables, namely x0 y0 th0, are used to store the reference pose.
% getPose function can be used to get measurements wrt the reference pose
% and evalPose function can be used to convert an absolute pose to one
% relative to the reference pose. The driveTo command accepts target pose
% wrt the reference pose. In order to disable all the functionality about
% the reference pose simply call resetOdo(socket,[0 0 0], [0 0 0]).
%
% When called with a single argument, namely the socket handle, resetOdo
% reads the current pose, and sets the reference pose so that the current
% pose will be [0 0 0] wrt it. This way of calling it effectively resets
% the odometry without making any change in the server.
%
% When called with two arguments, it again reads the current pose and pose0 
% argument defines how the current pose will be observed wrt the reference 
% pose. This function can be used to refine odometry when an estimation
% about the actual robot pose is made in world coordinates.
%
% When called with three arguments, it does not read the current pose, but
% instead it uses the curpose input, which is assumed to be a raw pose
% coming from the server. The rest is the same as the two argument case.

display('resetting odo');
global x0 y0 th0

if nargin<3
    curpose=getPose(socket,'abs');
end
if nargin>1
th0=curpose(3)-pose0(3);
diffxy=[cos(th0) -sin(th0); sin(th0) cos(th0)]*pose0(1:2)';
x0=curpose(1)-diffxy(1);
y0=curpose(2)-diffxy(2);
else
x0=curpose(1);
y0=curpose(2);
th0=curpose(3);    
end