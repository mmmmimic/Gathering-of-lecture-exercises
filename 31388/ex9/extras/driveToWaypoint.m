function [poseOut, poseCovOut, odoPose] = driveToWaypoint(mrcSck,poseIn, poseCovIn, odoPose,odoTargetPose, simulation)
% [poseOut, poseCovOut, odoPose] = DRIVETOWAYPOINT(mrcSck,poseIn,
% poseCovIn, odoPose,odoTargetPose, simulation)
% Drive to a given waypoint in odometry coordinates (odoTargetPose) while
% updating the pose estimate (poseIn, poseCovIn). The outputs are the
% updated pose estimate (poseOut, poseCovOut) and the final odometry pose
% (odoPose).

    global odoB kR kL
    global robotLinearSpeed robotAngularSpeed ts
    global realPose
   
    headingAngle = atan2(odoTargetPose(2)-odoPose(2),odoTargetPose(1)-odoPose(1));
    th1 = (mod(headingAngle-odoPose(3)+3*pi,2*pi)-pi);
    dist = sqrt((odoTargetPose(1)-odoPose(1))^2+(odoTargetPose(2)-odoPose(2))^2);
    th2 = (mod(odoTargetPose(3)-headingAngle+3*pi,2*pi)-pi);

    if(simulation)

        turn(th1);
        moveForward(dist);
        turn(th2);
        poseOut = poseIn;
        poseCovOut = poseCovIn;

    else
        
        resetOdo(mrcSck);
        resetOdoCov(mrcSck);

        ignoreText = 'ignoreobstacles';

        char(receiveMessage(mrcSck,0.01));

        if(abs(th1)<5*pi/180)
            relPose=subsPose(odoPose,odoTargetPose);
            relPose(3) = th1;
            odoTargetPose = addPose(odoPose,relPose)';
            command = sprintf('drive %f %f %f "rad" @v0.2: ($drivendist>%f)',[odoTargetPose dist]);
            mssendraw(mrcSck,uint8(['resetmotors' 10 ignoreText 10 command 10 'idle' 10]));
        else
            mssendraw(mrcSck,uint8(['resetmotors' 10 'turn ' num2str(th1*180/pi) ' @a0.1' 10  'resetmotors' 10 ignoreText 10 'fwd ' num2str(dist) ' @a0.1' 10]))
        end

        if(abs(th2)<5*pi/180)
           mssendraw(mrcSck,uint8(['syncevent "complete"' 10]));     
        else
           mssendraw(mrcSck,uint8(['resetmotors' 10 'turn ' num2str(th2*180/pi) ' @a0.1' 10 'syncevent "complete"' 10])); 
        end

        response = '';
        while(true)
            response = [response char(receiveMessage(mrcSck,0.01))];
            if(~isempty(strfind(response,'syncevent complete')))
                break
            end
        end

        [poseDiff, odoPose] = getPose(mrcSck);
        odoPose = odoPose';
        covDiff = getOdoCov(mrcSck);
        [poseOut, poseCovOut] = addPoseProbabilistic(poseIn,poseCovIn, poseDiff, covDiff);

    end

    function turn(angle)
        rotationIncrement = ts*robotAngularSpeed;
        remainingAngle = angle;
        while(remainingAngle)
            if(abs(remainingAngle) > rotationIncrement )
                delSr = rotationIncrement * odoB/2 * sign(remainingAngle);
                delSl = -rotationIncrement * odoB/2 * sign(remainingAngle);
                remainingAngle = remainingAngle - rotationIncrement*sign(remainingAngle);
            else
                delSr = remainingAngle * odoB/2;
                delSl = -remainingAngle * odoB/2;
                remainingAngle = 0;
            end
        
            varR = kR*abs(delSr);
            varL = kL*abs(delSl);

            delSr_ = delSr+randn*sqrt(varR); %Apply the noise on the measurements
            delSl_ = delSl+randn*sqrt(varL);

            [poseIn,poseCovIn] = odoUpdate(poseIn,poseCovIn, delSr, delSl);
            odoPose = odoUpdate(odoPose,zeros(3), delSr, delSl);
            realPose = odoUpdate(realPose,zeros(3), delSr_, delSl_);
       end
    end

    function moveForward(distance)
        forwardIncrement = ts*robotLinearSpeed;
        remainingDist = distance;
        while(remainingDist)
            if(remainingDist > forwardIncrement )
                delSr = forwardIncrement;
                delSl = forwardIncrement;
                remainingDist = remainingDist-forwardIncrement;
            else
                delSr = remainingDist;
                delSl = remainingDist;
                remainingDist = 0;
            end

            varR = kR*abs(delSr);
            varL = kL*abs(delSl);

            delSr_ = delSr+randn*sqrt(varR); %Apply the noise on the measurements
            delSl_ = delSl+randn*sqrt(varL);

            [poseIn,poseCovIn] = odoUpdate(poseIn,poseCovIn, delSr, delSl);
            odoPose = odoUpdate(odoPose,zeros(3), delSr, delSl);
            realPose = odoUpdate(realPose,zeros(3), delSr_, delSl_);
        end
    end
end