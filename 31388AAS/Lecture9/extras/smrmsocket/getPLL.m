function [ pose, absPose, Lines, ImWidth, ImHeight, scanData, success ] = getPLL( sckDemo,sckCam, sckLaser )

pose=[]; absPose=[]; Lines=[]; ImWidth=[]; ImHeight=[]; scanData=[]; success=0;

receiveMessage(sckDemo,0.02,0.02,100);
receiveImage(sckCam);
receiveMessage(sckCam,0.02,0.02,100);

if nargin > 2
    receiveMessage(sckLaser,0.02,0.02,100);
end

mssendraw(sckDemo,uint8(['eval $odox' 10]));
mssendraw(sckDemo,uint8(['eval $odoy' 10]));
mssendraw(sckDemo,uint8(['eval $odoth' 10]));
mssendraw(sckCam,uint8(['hough device=10 parametric' 10]));

if nargin >2
    mssendraw(sckLaser,uint8(['scanget codex=TAG' 10]));
end

pause(0.2);

header=char(msrecvraw(sckCam,57,1))';

if length(header)~=57
    display('Couldnt get lines header');    
    return;
end

if length(header)<10
    display(['Couldnt get header']);
    return;
end

if strcmp(header(1:10),'NoOfLines:')==1
   NoOfLines = sscanf(header(1:15),'NoOfLines: %d');
   BytesPerLine = sscanf(header(16:32), ' BytesPerLine: %d');
   ImWidth = sscanf(header(34:44),'Width: %d');
   ImHeight = sscanf(header(46:end), 'Height: %d');
else
    display(['Couldnt resolve header: ' header]);
    return;
end

messageSize=NoOfLines*BytesPerLine;

rest=char(msrecvraw(sckCam,messageSize,1)');

lineData = sscanf(rest, ' %f %f %f %f');

Lines=reshape(lineData,4, length(lineData)/4)';


if size(Lines,1)~=NoOfLines || size(Lines,1)==0
    display('Couldnt get lines');    
    return;
end

response=char(receiveMessage(sckDemo,0.1,0.02));
pose=sscanf(response,'%f')';

if length(pose)~=3
    display('failed to get pose');
    return
end

absPose=pose;

pose=evalPose(pose);

if nargin > 2
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
end

success=1;