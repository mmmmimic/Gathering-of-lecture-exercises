clear all

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
    if(mod(iter,plotNth) == 0 )
        %figure(1)
        %%
        subplot(2,1,1);
        %%
        plot(poses(1,1:iter)',poses(2,1:iter)','r');
        hold on
        h1=plot(realPoses(1,1:iter)',realPoses(2,1:iter)','b');
        plot(worldPoints(1,:)',worldPoints(2,:),'b-');    
        %arrow(pose(1:2),pose(1:2)+[cos(pose(3));sin(pose(3))]/5,10,[],[],[],'FaceColor','r');
        %arrow(realPose(1:2),realPose(1:2)+[cos(realPose(3));sin(realPose(3))]/5,10,[],[],[],'FaceColor','b');
         plot([pose(1) pose(1)+cos(pose(3))/5],[pose(2) pose(2)+sin(pose(3))/5],'r');
         plot(pose(1)+cos(pose(3))/5,pose(2)+sin(pose(3))/5,'ro')
         plot([realPose(1) realPose(1)+cos(realPose(3))/5],[realPose(2) realPose(2)+sin(realPose(3))/5],'b');
         plot(realPose(1)+cos(realPose(3))/5,realPose(2)+sin(realPose(3))/5,'bo')
        for ellipsoidIndex = [1:plotNthEllipsoid:iter iter]
            h=plot_gaussian_ellipsoid(poses(1:2,ellipsoidIndex), poseCovs{1,ellipsoidIndex}(1:2,1:2));
            set(h,'color','r');
            plot(poses(1,ellipsoidIndex),poses(2,ellipsoidIndex),'rx')
        end
        legend([h1,h], 'Real Path', 'Estimated Path')
        title('Real and the estimated paths of the robot')
        xlabel('x(m)');
        ylabel('y(m)');
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
                
        %Extract lines from the scan data
        laserLines = ransacLines(scanCar, [maxNoOfLines noOfRandomCouples distThreshold minLineSupport minNoOfPoints]);
        noOfLaserLines = size(laserLines,2);

        % Plot the predicted lines and the laser scanner data line
        % prediction is performed by your script named projectToLaser
        if(plotLaserDataWithPredictedLines)
            %figure(2)
            %%
            subplot(2,1,2);
            %%
            h1=plot(scanCar(1,:)',scanCar(2,:),'.r');
            hold on
            legend(h1, 'Data Points');
            for worldIndex = 1:noOfWorldLines %Plot the predicted lines
                [projLine lineCov]=projectToLaser(worldLines(1:2,worldIndex),pose,poseCov);
                h2=plotLine(projLine,'b');
            end
            if(noOfWorldLines>=1)
                legend([h1 h2], 'Data Points', 'Projected Lines');
            end
            if(plotExtractedLines)
                for lsrIndex = 1:noOfLaserLines % Plot the extracted lines
                    h3=plotLine(laserLines(:,lsrIndex),'r');            
                end
                if(noOfLaserLines>=1)
                    legend([h1 h2 h3], 'Data Points', 'Projected Lines', 'Extracted lines');
                end
            end
            title('Laser scan along with the predicted and the extracted lines, ')
            xlabel('x(m)');
            ylabel('y(m)');
            axis([0 2 -2 2]);
            hold off
        end
            realPose    
        %Plot the estimated and the extracted lines in the parameter space
        if(plotLineParameters)
            figure(3)
            plot(laserLines(1,:),laserLines(2,:),'rx');
            hold on
            for lsrIndex = 1:noOfLaserLines
                h = plot_gaussian_ellipsoid(laserLines(:,lsrIndex),[varAlpha 0; 0 varR]);
                set(h,'color','r');
            end

            for worldIndex = 1:noOfWorldLines
                [projLine lineCov]=projectToLaser(worldLines(1:2,worldIndex),pose,poseCov);
                plot(projLine(1),projLine(2),'x');
                plot_gaussian_ellipsoid(projLine,lineCov);
            end
            axis([-pi,pi,0,squareWidth]);
            title('the predicted and the extracted lines in the line parameter space')
            xlabel('alpha')
            ylabel('r(m)')
            hold off
        end
        pause(1)
    end   
            
end