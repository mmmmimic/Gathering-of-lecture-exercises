function [ poseOut, poseCovOut ] = measurementUpdate( poseIn, poseCovIn, matchResult )
%[ poseOut, poseCovOut ] =MEASUREMENTUPDATE ( poseIn, poseCovIn,
%matchResult ) perform extended Kalman filter measurement update on the
%estimated robot pose poseIn with covariance poseCovIn using a set of
%matched predicted and extracted laser scanner lines given in matchResult.
%The arguments are defined as:
%       poseIn: The estimated robot pose given as [x,y,theta]
%       poseCovIn: The estimated covariance matrix of the robot pose
%       matchResult: A (5xnoOfWorldLines) matrix whose columns are
%       individual pairs of line matches. It is structured as follows:
%       matchResult = [ worldLine(1,1) , worldLine(1,2) ...  ]
%                     [ worldLine(2,1) , worldLine(2,2)      ]
%                     [ innovation1(1) , innovation2(1)      ]
%                     [ innovation1(2) , innovation2(2)      ]
%                     [ matchIndex1    , matchIndex2    ...  ]
%           Note that the worldLines are in the world coordinates!
%
%       poseOut: The updated robot pose estimate
%       poseCovOut: The updated estimate of the robot pose covariance
%       matrix

% Constants
% The laser scanner pose in the robot frame is read globally(lsrRelpose)
% The varAlpha and varR are the assumed variances of the parameters of
% the extracted lines, they are also read globally
global lsrRelPose varAlpha varR
x_l = lsrRelPose(1);
y_l = lsrRelPose(1);
theta = poseIn(3);
matchResult = matchResult(:,(matchResult(5,:)>0));
N = size(matchResult,2);% We have N matches
H = zeros(2*N, 3);
sigma_R = zeros(2*N, 2*N);
v_t = zeros(2*N,1);
for i = 1:N
    alpha_w = matchResult(1,i);
    H(i*2-1:i*2,:) = [0,0,-1;-cos(alpha_w), -sin(alpha_w), -x_l*sin(alpha_w-theta)+y_l*cos(alpha_w-theta)];
    sigma_R(i*2-1:i*2,i*2-1:i*2) = [varAlpha,0;0,varR];
    v_t(i*2-1:i*2) = matchResult(3:4,i);
end
spy(H);
spy(sigma_R);
sigma_in = H*poseCovIn*H'+sigma_R;
% Kalman gain
K_t = poseCovIn*H'*inv(sigma_in);
% update the pose
%poseOut = poseIn;
poseOut = poseIn+K_t*v_t;
% update the covariance
%poseCovOut = poseCovIn;
poseCovOut = poseCovIn-K_t*sigma_in*K_t';

end
