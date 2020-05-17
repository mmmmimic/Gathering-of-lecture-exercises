%% World constants
% The box
global lines
squareWidth = 1.84; % The width of the box that the robot is in
lines = [[1;1;1;-1],[1;-1;-1;-1],[-1;-1;-1;1],[-1;1;1;1]]*squareWidth/2; % Lines describing the box
worldPoints = [lines(1:2,:) lines(1:2,1)]; % Corner points of the box
noOfWorldLines = size(lines,2);
worldLines =convertToPolar(lines); % The lines in the (alpha, rho) parameterization

%% Real Robot constants
global odoB kR kL
odoB = 0.3; % The separation between the two wheels
kR = 0.0001; % The variance of the linear uncertainty for a 1m move for the right wheel
kL = 0.0001; % The variance of the linear uncertainty for a 1m move for the left wheel
smrNo=11
%% Kalman filter Robot constants
global odoB_kf kR_kf kL_kf
odoB_kf = 0.3; % The separation between the two wheels
kR_kf = 0.0001; % The variance of the linear uncertainty for a 1m move for the right wheel
kL_kf = 0.0001; % The variance of the linear uncertainty for a 1m move for the left wheel

%% Simulation constants
global ts robotLinearSpeed robotAngularSpeed robotPathWidth robotPathRadius trackNo
noOfIter = 42; %The number of simulation iterations, if it's for N times round square, noOfIter=N*4+2
ts = 0.01; %The time period of each iteration
plotNth = 100; %Plot the pose data every nth iteration
plotNthEllipsoid = 100; %Plot a gaussian ellipsoid for every nth pose
simulation=true;
%% Path constants
trackNo = 3; %The path followed by the robot, 1: line, 2: circle, 3: square
robotLinearSpeed = 0.3; %The linear speed of the robot for path segments that it is moving forward
robotAngularSpeed = 1; %The angular speed of the robot fot path segments that it is turning
robotPathWidth = squareWidth - 0.6; % For the square path, the width of the square
robotPathRadius = 1; % For the circular path, the radius of the circle

%% Laser scanner constants
global varAlpha varR
global lsrRelPose
lsrPer = 100; %The period of laser scans
lsrRelPose = [0.28,0,0]; %The pose of the laser scanner in the robot frame
lsrRelRot = [cos(lsrRelPose(3)) sin(lsrRelPose(3)); -sin(lsrRelPose(3)) cos(lsrRelPose(3))];
varAlpha = 0.001; %The assumed variance on the orientation of a line extracted from the laser scanner
varR = 0.0004; %%The assumed variance on the distance of a line extracted from the laser scanner

plotLaserDataWithPredictedLines = true;
plotExtractedLines = false;
plotLineParameters = false;

plotMatchedLines = false;
plotAfterMeasurementUpdate = true;


%% RANSAC constants
maxNoOfLines = 50; % The maximum number of line extraction iterations
noOfRandomCouples = 20; % The number of random point couples to try before choosing the best candidate
distThreshold=0.01; % The distance threshold determining whether a point is supporting a line
minLineSupport=20; % The minimum number of points to support a line for the line to be accepted
minNoOfPoints=20; % The minimum number of points to continue line extraction iterations


%% Initials
pose = [-squareWidth/2+0.1;-squareWidth/2+0.1;0]; % The !!estimated!! initial pose of the robot

poseCov = [0.0001 0 0  % The assumed uncertainty on the initial pose
           0 0.0001 0
           0 0 0.0001];
           
       