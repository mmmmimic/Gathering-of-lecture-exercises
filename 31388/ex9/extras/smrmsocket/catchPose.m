function [ pose, absPose ] = catchPose( sckDemo )
pose=[];

while(length(pose)<3)
    response=char(receiveMessage(sckDemo,0.1,0.1));
    pose=[pose sscanf(response,'%f')']; %#ok<AGROW>
end

absPose=pose;

pose=evalPose(pose);