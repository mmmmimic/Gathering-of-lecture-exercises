function driveTo( socket, tP, v)
% function driveTo( socket, tP )
%
% Drive to the given target pose. The target pose is assumed to be wrt the
% reference pose. To read more about the reference pose see resetOdo
% function.
%
% socket: the handle object for the socket connection
% tP: The target pose

if(nargin==3)
   velocity = sprintf('%f',v); 
else
    velocity ='0.2';
end

global x0 y0 th0

if isempty(x0) || isempty(y0) ||isempty(th0) 
    resetOdo(socket,[0 0 0], [0 0 0]);
end

target=[([cos(th0) -sin(th0); sin(th0) cos(th0)]*[tP(1) tP(2)]')'+[x0 y0] th0+tP(3)];
command = sprintf(['drive %f %f %f "rad" @v' velocity ': (1)'],target);

mssendraw(socket,uint8([command 10]));
