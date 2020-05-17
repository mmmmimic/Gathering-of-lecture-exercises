function [success] = driveTowardsWaypoint(mrcSck,curPose,targetPose, ignoreObstacles,timeOut)

headingAngle = atan2(targetPose(2)-curPose(2),targetPose(1)-curPose(1));
th1 = 180/pi*(mod(headingAngle-curPose(3)+3*pi,2*pi)-pi);
dist = sqrt((targetPose(1)-curPose(1))^2+(targetPose(2)-curPose(2))^2);
th2 = 180/pi*(mod(targetPose(3)-headingAngle+3*pi,2*pi)-pi);

if(ignoreObstacles)
    ignoreText = 'ignoreobstacles';
else
    ignoreText = '';
end

char(receiveMessage(mrcSck,0.01));

if(abs(th1)<5)
    odoPose=getPose(mrcSck,'abs');
    relPose=subsPose(curPose,targetPose);
    relPose(3) = th1*pi/180;
    odoTargetPose = addPose(odoPose,relPose)';
    command = sprintf('drive %f %f %f "rad" @v0.2: ($drivendist>%f)',[odoTargetPose dist]);
    mssendraw(mrcSck,uint8(['resetmotors' 10 ignoreText 10 command 10 'idle' 10]));
else
    mssendraw(mrcSck,uint8(['resetmotors' 10 'turn ' num2str(th1) ' @a0.1' 10  'resetmotors' 10 ignoreText 10 'fwd ' num2str(dist) ' @a0.1' 10]))
end

if(abs(th2)<5)
   mssendraw(mrcSck,uint8(['syncevent "complete"' 10]));     
else
   mssendraw(mrcSck,uint8(['resetmotors' 10 'turn ' num2str(th2) ' @a0.1' 10 'syncevent "complete"' 10])); 
end
