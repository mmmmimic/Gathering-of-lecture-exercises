function [ pose, absPose, scanData, success ] = getPosenLaser( sckDemo, sckLaser )

pose=[]; absPose=[]; scanData=[]; success=0;

receiveMessage(sckDemo,0.02);
receiveMessage(sckLaser,0.02);

mssendraw(sckDemo,uint8(['eval $odox' 10]));
mssendraw(sckDemo,uint8(['eval $odoy' 10]));
mssendraw(sckDemo,uint8(['eval $odoth' 10]));
mssendraw(sckLaser,uint8(['scanget codex=TAG' 10]));

pause(0.2);

pose=[];

while(length(pose)<3)
    response=char(receiveMessage(sckDemo,0.1,0.1));
    pose=[pose sscanf(response,'%f')']; %#ok<AGROW>
end

absPose=pose;

pose=evalPose(pose);

response=char(receiveMessage(sckLaser,0.1,0.02,15));
nlIndices=find(response==10);
if length(nlIndices)<4
   display('failed to get laser scan');
   return
end
beginIndex = nlIndices(4)+1;
endIndex=nlIndices(end-1)-1;
scanData=sscanf(response(beginIndex:endIndex),['<lval f="%f" ang="%f" dist="%f"/>' char(10)]);
scanData=scanData(1:floor(length(scanData)/3)*3);
scanData=reshape(scanData,3,length(scanData)/3);

success=1;