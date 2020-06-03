function [ poseOut, covOut ] = positionPrediction( poseIn,covIn,delSr,delSl)
%[poseOut, covOut] = POSITIONPREDICTION(poseIn,covIn,delSr,delSl) perform
%one step of robot pose prediction from a set of wheel displacements
%   poseIn = old robot pose
%   covIn = uncertainty on the old robot pose
%   delSr = right wheel linear displacement
%   delSl = left wheel linear displacement


%% Constants
% The robot parameters are read globally, odoB is the wheel separation, kR
% and kL are the odometry uncertainty parameters
global odoB kR kL 

%% pose update

%poseOut = [0;0;0];
wheel_sub = delSr-delSl;
wheel_sum = delSr+delSl;
dx = wheel_sum/2*cos(poseIn(3)+wheel_sub/(2*odoB));
dy = wheel_sum/2*sin(poseIn(3)+wheel_sub/(2*odoB));
dtheta = wheel_sub/odoB;
theta = poseIn(3);
poseOut = poseIn+[dx;dy;dtheta];
poseOut(3) = norm_ang(poseOut(3)); % commend it if nomalization is not required
%% Covariance update
               
Qt = [kR*abs(delSr),0;0,kL*abs(delSl)];
ds = sqrt(dx^2+dy^2);
the = theta+dtheta/2;
Fx = [1,0,-ds*sin(the);
    0,1,ds*cos(the);
0,0,1];
Fu = [1/2*cos(the)-ds/(2*odoB)*sin(the),1/2*cos(the)+ds/(2*odoB)*sin(the);
    1/2*cos(the)-ds/(2*odoB)*cos(the),1/2*cos(the)+ds/(2*odoB)*cos(the);
    1/odoB,-1/odoB];
covOut = Fx*covIn*Fx'+Fu*Qt*Fu';
%covOut = zeros(3,3);

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