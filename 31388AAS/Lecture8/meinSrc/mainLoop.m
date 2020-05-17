% Get a laser scan
success = 0;
while(~success)
    [scan success] = getLaser(lsrSck,simulation);
end
if iter>1
    pose = pose1;
    poseCov = poseCov1;
end
%scan is in polar coordinates, convert it to cartesian coordinates
scanCar = [(cos(scan(1,:)).*scan(2,:));(sin(scan(1,:)).*scan(2,:))];

%Extract lines from the scan data
laserLines = ransacLines(scanCar, [maxNoOfLines noOfRandomCouples distThreshold minLineSupport minNoOfPoints]);

%Perform line matching
matchResult = match(pose, poseCov,worldLines, laserLines);
matchPose = pose; %Recording the pose during mathing for plotting purposes

%Update the pose estimate using the measured lines
[pose, poseCov] = measurementUpdate(pose,poseCov, matchResult);
mPose = pose;
%% Move to the origin(0,0)
% Find the next waypoint in world coordinates and put it in targetPose
%targetPose = [0; 0; 0];
%Find the transformation from the estimated world coordinates to the
%odometry coordinates
transform = findTransform(odoPose, pose);

corners = [0.5,0.5,-0.5,-0.5;-0.5,0.5,0.5,-0.5;pi/2,pi,-pi/2,0];


%Find the target pose in the odometry coordinates
odoTargetPose = trans(transform,targetPose);

targetPose = corners(:,mod(iter, 4)+1);

%Drive to the waypoint while updating the pose estimation
[pose1, poseCov1, odoPose] = driveToWaypoint(mrcSck, pose, poseCov, odoPose, odoTargetPose,simulation);