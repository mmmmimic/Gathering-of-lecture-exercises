clear all

addpath([pwd filesep 'extras'])
addpath([pwd filesep 'yourScripts'])
addpath([pwd filesep 'extras' filesep 'arrow'])
addpath([pwd filesep 'extras' filesep 'gaussianEllipsoid'])
addpath([pwd filesep 'extras' filesep 'smrmsocket'])

global realPose

%% Constants
constants % Calling the script with the constants

%% Run
if(simulation)
    realPose = pose + randn(3,1).*sqrt([poseCov(1,1); poseCov(2,2); poseCov(3,3)]); %Generate a real pose based on the initial uncertainty
    odoPose = [0;0;0];
    lsrSck = 0;
    mrcSck = 0;
else
    smrName = ['smr' int2str(smrNo) '.iau.dtu.dk'];

    mrcSck = msconnect(smrName,31001);
    lsrSck = msconnect(smrName,24919);
    if(mrcSck == -1 || lsrSck == -1)
        display(['One of the connections failed; mrcSck:' int2str(mrcSck) ' lsrSck:' int2str(lsrSck)] );
        return
    end
    pause(5);
    display('starting');

    odoPose = getPose(mrcSck,'abs')';
end
    
%Generate the arrays for logging
poses = zeros(3,noOfIter+1);
poseCovs = cell(1,noOfIter+1);
realPoses = zeros(3,noOfIter+1);
odoPoses = zeros(3,noOfIter+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%corners = [0.5,0.5,-0.5,-0.5;-0.5,0.5,0.5,-0.5;pi/2,pi,-pi/2,0];
%corners = [0.9,0.9,-0.9,-0.9;-0.9,0.9,0.9,-0.9;pi/2,pi,-pi/2,0];
targetPose = [0;0;0];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:noOfIter+1 %+1 is because that the first point is not [0,0,0], we need to move to the origin first
    
    poses(:,iter) = pose;
    poseCovs{1,iter} = poseCov;
    if(simulation)
        realPoses(:,iter) = realPose;
    end
    odoPoses(:,iter) = odoPose;
    
    figure(1);
    handles = [];
    handles(1)=plot(poses(1,1:iter)',poses(2,1:iter)','r');
    hold on
    if(simulation)
        handles(2)=plot(realPoses(1,1:iter)',realPoses(2,1:iter)','b');
        arrow(realPose(1:2),realPose(1:2)+[cos(realPose(3));sin(realPose(3))]/5,10,[],[],[],'FaceColor','b');
        title('The real and the estimated paths of the robot')
    else
        title('The estimated path of the robot')
    end
    plot(worldPoints(1,:)',worldPoints(2,:),'k-');    
    arrow(pose(1:2),pose(1:2)+[cos(pose(3));sin(pose(3))]/5,10,[],[],[],'FaceColor','r');
    h=plot_gaussian_ellipsoid(poses(1:2,iter), poseCovs{1,iter}(1:2,1:2));
    set(h,'color','r');
    
    xlabel('x(m)');
    ylabel('y(m)');
    legend(handles, 'Estimated path','Actual path');
    hold off
    
    figure(2);
    plot(odoPoses(1,1:iter)',odoPoses(2,1:iter)','r');
    hold on
    plot(worldPoints(1,:)',worldPoints(2,:),'k-');    
    hold off
    title('Pure odometry')
    pause(0.01)
    
    if(iter == noOfIter+1)
        break
    end

    mainLoop

    noOfPairs = sum(matchResult(5,:)>0);

    colors = 'brkm';
    figure(3);
    plot(scanCar(1,:)',scanCar(2,:)','.');
    hold on
    handles = [];
    for matchIndex = find(matchResult(5,:)>0)
        [projLine lineCov]=projectToLaser(matchResult(1:2,matchIndex),matchPose,poseCov);
        handles(1) = plotLine(projLine,colors(matchIndex),'--');
        handles(2) = plotLine(laserLines(:,matchResult(5,matchIndex)),colors(matchIndex));            
    end
    title('The predicted and the extracted lines, ')
    xlabel('x(m)');
    ylabel('y(m)');
    axis([0 2 -2 2]);
    legend(handles, 'Predicted lines','Measured lines');
    hold off
    %targetPose = corners(:,mod(iter, 4)+1);
end