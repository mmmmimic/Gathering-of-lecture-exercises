function [ poseOut, covOut ] = odoUpdate( poseIn,covIn,delSr,delSl)
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
D = (delSr-delSl)/odoB;
L = (delSr+delSl)/2;
th_ = poseIn(3)+D/2;
poseOut = poseIn + [L*cos(th_);L*sin(th_);D];

%% Covariance update

delPnew_delY=[  cos(th_)/2-L*sin(th_)/odoB,    cos(th_)/2+L*sin(th_)/odoB
                sin(th_)/2+L*cos(th_)/odoB,    sin(th_)/2-L*cos(th_)/odoB
                1/odoB,                        -1/odoB];

delPnew_delPold = [ 1,    0,      -L*sin(th_)
                    0,    1,      L*cos(th_)
                    0,    0,      1];

covU = [kR*abs(delSr),    0
        0,                  kL*abs(delSl)];
                
covOut = delPnew_delPold*covIn*delPnew_delPold'+delPnew_delY*covU*delPnew_delY';


end