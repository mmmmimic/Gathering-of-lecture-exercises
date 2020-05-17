function [delSr, delSl] = simulateWorld(index)

global realPose
global odoB kR kL
global ts

global task

global robotLinearSpeed robotAngularSpeed robotPathWidth robotPathRadius trackNo

global remainingDist remainingAngle

firstCorner = [-robotPathWidth/2 -robotPathWidth/2];

rotationIncrement = ts*robotAngularSpeed;
forwardIncrement = ts*robotLinearSpeed;

if(index == 0) %Initialization
    switch(trackNo)
        case 1
            task = 5;
        case 2
            task = 6;
        case 3
            if(all(realPose(1:2) == firstCorner'))
                task = 3;
                remainingAngle = -realPose(3);
            else
                task = 1;
            end
    end
    return
end

switch(task)
    case 1 % Rotate towards first corner
        angleToCorner = atan2(firstCorner(2)-realPose(2),firstCorner(1)-realPose(1)) - realPose(3);
        if(abs(angleToCorner) > rotationIncrement )
            delSr = rotationIncrement * odoB/2 * sign(angleToCorner);
            delSl = -rotationIncrement * odoB/2 * sign(angleToCorner);
        else
            delSr = angleToCorner * odoB/2;
            delSl = -angleToCorner * odoB/2;
            task = 2;
        end
    case 2 % Move towards first corner
        distToCorner = sqrt((firstCorner(1)-realPose(1))^2+(firstCorner(2)-realPose(2))^2);
        if(distToCorner > forwardIncrement )
            delSr = forwardIncrement;
            delSl = forwardIncrement;
        else
            delSr = distToCorner;
            delSl = distToCorner;
            task = 3;
            remainingAngle = -realPose(3);
        end
    case 3 % Rotate towards next edge
        if(abs(remainingAngle) > rotationIncrement )
            delSr = rotationIncrement * odoB/2 * sign(remainingAngle);
            delSl = -rotationIncrement * odoB/2 * sign(remainingAngle);
            remainingAngle = remainingAngle - rotationIncrement*sign(remainingAngle);
        else
            delSr = remainingAngle * odoB/2;
            delSl = -remainingAngle * odoB/2;
            task = 4;
            remainingDist = robotPathWidth;
        end
        
    case 4 % Move along edge

        if(remainingDist > forwardIncrement )
            delSr = forwardIncrement;
            delSl = forwardIncrement;
            remainingDist = remainingDist-forwardIncrement;
        else
            delSr = remainingDist;
            delSl = remainingDist;
            task = 3;
            remainingAngle = pi/2;
        end
    case 5 % Move on a straight line
        
        delSr = forwardIncrement;
        delSl = forwardIncrement;
        
    case 6 % Move along a circle
        
        rotation = forwardIncrement/robotPathRadius;
        delSr = forwardIncrement + rotation*odoB/2;
        delSl = forwardIncrement - rotation*odoB/2;
        
end

D = (delSr - delSl) / odoB;
L = (delSr + delSl) / 2;


realPose = realPose + [L*cos(realPose(3)+D/2); L*sin(realPose(3)+D/2);D];

varR = kR*abs(delSr);
varL = kL*abs(delSl);

delSr = delSr+randn*sqrt(varR); %Apply the noise on the measurements
delSl = delSl+randn*sqrt(varL);

end