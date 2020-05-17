function [scan, success ] = getLaser( sckLaser, simulation )

global realPose lsrRelPose lines

if(simulation)
    
    lsrRealPose = addPose(realPose,lsrRelPose); 
    scan = laserscan(lsrRealPose(1),lsrRealPose(2),lsrRealPose(3),lines,5,180/600);
    success=1;
else
    success=0;

    receiveMessage(sckLaser,0.02);

    mssendraw(sckLaser,uint8(['scanget codex=TAG' 10]));

    pause(0.2);

    response=char(receiveMessage(sckLaser,0.1,0.02,15));
    nlIndices=find(response==10);
    if length(nlIndices)<4
       display('failed to get laser scan');
       scan = [];
       return
    end
    beginIndex = nlIndices(4)+1;
    endIndex=nlIndices(end-1)-1;
    scanData=sscanf(response(beginIndex:endIndex),['<lval f="%f" ang="%f" dist="%f"/>' char(10)]);
    scanData=scanData(1:floor(length(scanData)/3)*3);
    scanData=reshape(scanData,3,length(scanData)/3);

    scan = [scanData(2,:)*pi/180; scanData(3,:)];
    
    success=1;
end

