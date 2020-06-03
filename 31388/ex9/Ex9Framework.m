clear all; close all; clc;

addpath([pwd filesep 'extras'])
addpath([pwd filesep 'yourScripts'])
addpath([pwd filesep 'extras' filesep 'arrow'])
addpath([pwd filesep 'extras' filesep 'gaussianEllipsoid'])

global realPose

%% Constants
constants % Calling the script with the constants

%% Run
realPose = pose + randn(3,1).*sqrt([poseCov(1,1); poseCov(2,2); poseCov(3,3)]); %Generate a real pose based on the initial uncertainty

simulateWorld(0); % Initialize simulateWorld

%Generate the arrays for logging
poses = zeros(3,noOfIter);
poseCovs = cell(1,noOfIter);
realPoses = zeros(3,noOfIter);

for iter = 1:noOfIter
    % Move a bit
    [delSr, delSl]=simulateWorld(iter); %delSr and delSl are the left and right wheel displacements

    %Your script performing position prediction
    [pose, poseCov] = positionPrediction(pose, poseCov, delSr, delSl);
    
    poses(:,iter) = pose;
    poseCovs{1,iter} = poseCov;
    realPoses(:,iter) = realPose;
    
    %Plot the real path along with the predicted path
    if(mod(iter,plotNth) == 0 || mod(iter,lsrPer) == 0)
        figure(1)
        handles = [];
        handles(1)=plot(poses(1,1:iter)',poses(2,1:iter)','r');
        hold on
        handles(2)=plot(realPoses(1,1:iter)',realPoses(2,1:iter)','b');
        plot(worldPoints(1,:)',worldPoints(2,:),'k-');    
        arrow(pose(1:2),pose(1:2)+[cos(pose(3));sin(pose(3))]/5,10,[],[],[],'FaceColor','r');
        arrow(realPose(1:2),realPose(1:2)+[cos(realPose(3));sin(realPose(3))]/5,10,[],[],[],'FaceColor','b');
        for ellipsoidIndex = [1:plotNthEllipsoid:iter iter]
            h=plot_gaussian_ellipsoid(poses(1:2,ellipsoidIndex), poseCovs{1,ellipsoidIndex}(1:2,1:2));
            set(h,'color','r');
            plot(poses(1,ellipsoidIndex),poses(2,ellipsoidIndex),'rx')
        end
        title('Real and the estimated paths of the robot')
        xlabel('x(m)');
        ylabel('y(m)');
        legend(handles, 'Estimated path','Actual path');
        hold off
        pause(0.01)
    end

    
    %get a scan if the laser scan period has passed
    if(mod(iter,lsrPer) == 0)

        % Calculate the real pose of the laser scanner to use with the 
        %laserscan function. Note that addPose is a coordinate
        %transformation, not a vector addition
        lsrRealPose = addPose(realPose,lsrRelPose); 
        scan = laserscan(lsrRealPose(1),lsrRealPose(2),lsrRealPose(3),lines,5,1);
        
        %scan is in polar coordinates, convert it to cartesian coordinates
        scanCar = [(cos(scan(1,:)).*scan(2,:));(sin(scan(1,:)).*scan(2,:))];
       % figure(3)
        % plot(scanCar(1,:),scanCar(2,:),'r.');
                
        %Extract lines from the scan data
        laserLines = ransacLines(scanCar, [maxNoOfLines noOfRandomCouples distThreshold minLineSupport minNoOfPoints]);
        noOfLaserLines = size(laserLines,2);
        
        % Plot the predicted and the extracted lines
        figure(2)
        clf reset
        subplot(121)
        hold on
        handles = [];
        for worldIndex = 1:noOfWorldLines
            [projLine lineCov]=projectToLaser(worldLines(1:2,worldIndex),pose,poseCov);
            handles(1) = plotLine(projLine,'b');
        end

        for lsrIndex = 1:noOfLaserLines
            handles(2) = plotLine(laserLines(:,lsrIndex),'r');
            legend(handles, 'Predicted lines','Measured lines');
        end
        title('The predicted and the extracted lines, ')
        xlabel('x(m)');
        ylabel('y(m)');
        axis([0 2 -2 2]);
        hold off

        subplot(122)
        plot(laserLines(1,:),laserLines(2,:),'rx');
        hold on
        handles = [];
        for worldIndex = 1:noOfWorldLines
            [projLine lineCov]=projectToLaser(worldLines(1:2,worldIndex),pose,poseCov);
            plot(projLine(1),projLine(2),'x');
            h = plot_gaussian_ellipsoid(projLine,lineCov);
            handles(2) = h;
        end
        
        for lsrIndex = 1:noOfLaserLines
            h = plot_gaussian_ellipsoid(laserLines(:,lsrIndex),[varAlpha 0; 0 varR]);
            set(h,'color','r');
            handles(1)=h;
            legend(handles, 'Predicted lines','Measured lines');
        end
        
        axis([-pi,pi,0,squareWidth]);
        title('the predicted and the extracted lines in the line parameter space')
        xlabel('alpha')
        ylabel('r(m)')
        hold off
        pause
            
        matchResult = match(pose, poseCov,worldLines, laserLines);
        noOfPairs = sum(matchResult(5,:)>0);

        if(plotMatchedLines)
            colors = 'brkm';
            figure(2)
            clf reset
            subplot(121)
            hold on
            handles = [];
            for matchIndex = find(matchResult(5,:)>0)
                [projLine lineCov]=projectToLaser(matchResult(1:2,matchIndex),pose,poseCov);
                handles(1) = plotLine(projLine,colors(matchIndex),'--');
                handles(2) = plotLine(laserLines(:,matchResult(5,matchIndex)),colors(matchIndex));            
            end
            title('The predicted and the extracted lines with matched colors, ')
            xlabel('x(m)');
            ylabel('y(m)');
            axis([0 2 -2 2]);
            legend(handles, 'Predicted lines','Measured lines');
            hold off
            
            subplot(122)
            hold on
            handles = [];
            for matchIndex = find(matchResult(5,:)>0)
                h = plot_gaussian_ellipsoid(laserLines(:,matchResult(5,matchIndex)),[varAlpha 0; 0 varR]);
                set(h,'color',colors(matchIndex));
                handles(2) =h;
                plot(laserLines(1,matchResult(5,matchIndex)),laserLines(2,matchResult(5,matchIndex)),[colors(matchIndex) 'x']);
                [projLine lineCov]=projectToLaser(matchResult(1:2,matchIndex),pose,poseCov);
                plot(projLine(1),projLine(2),[colors(matchIndex) 'o']);
                h = plot_gaussian_ellipsoid(projLine,lineCov);
                set(h,'color',colors(matchIndex),'linestyle','--');
                handles(1) = h;
                legend(handles, 'Predicted lines','Measured lines');
            end
            axis([-pi,pi,0,squareWidth]);
            title('the predicted and the extracted lines in the line parameter space')
            xlabel('alpha')
            ylabel('r(m)')
            hold off
            pause
        end

        
        [pose, poseCov] = measurementUpdate(pose,poseCov, matchResult);

        if(plotAfterMeasurementUpdate)
            colors = 'brkm';
            figure(2)
            clf reset
            subplot(121)
            hold on
            for matchIndex = find(matchResult(5,:)>0)
                [projLine lineCov]=projectToLaser(matchResult(1:2,matchIndex),pose,poseCov);
                plotLine(projLine,colors(matchIndex),'--');
                plotLine(laserLines(:,matchResult(5,matchIndex)),colors(matchIndex));            
            end
            title('The predicted and the extracted lines after measurement update, ')
            xlabel('x(m)');
            ylabel('y(m)');
            axis([0 2 -2 2]);
            hold off
            
            subplot(122)
            hold on
            for matchIndex = find(matchResult(5,:)>0)
                h = plot_gaussian_ellipsoid(laserLines(:,matchResult(5,matchIndex)),[varAlpha 0; 0 varR]);
                set(h,'color',colors(matchIndex));
                plot(laserLines(1,matchResult(5,matchIndex)),laserLines(2,matchResult(5,matchIndex)),[colors(matchIndex) 'x']);
                [projLine lineCov]=projectToLaser(matchResult(1:2,matchIndex),pose,poseCov);
                plot(projLine(1),projLine(2),[colors(matchIndex) 'o']);
                h = plot_gaussian_ellipsoid(projLine,lineCov);
                set(h,'color',colors(matchIndex),'linestyle','--');
            end
            axis([-pi,pi,0,squareWidth]);
            title('the predicted and the extracted lines in the line parameter space')
            xlabel('alpha')
            ylabel('r(m)')
            hold off
            
            figure(1)
            hold on
            arrow(pose(1:2),pose(1:2)+[cos(pose(3));sin(pose(3))]/5,10,[],[],[],'FaceColor','m');
            hold off
            
            pause
        end
        
    end   

end
