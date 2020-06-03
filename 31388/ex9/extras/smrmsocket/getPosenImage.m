function [ pose, absPose, image, scanData ] = getPosenImage( sckDemo,sckCam, sckLaser )


receiveMessage(sckDemo,0.02,0.02,100);
receiveImage(sckCam);
receiveMessage(sckCam,0.02,0.02,100);

if nargin > 2
    receiveMessage(sckLaser,0.02,0.02,100);
end

mssendraw(sckDemo,uint8(['eval $odox' 10]));
mssendraw(sckDemo,uint8(['eval $odoy' 10]));
mssendraw(sckDemo,uint8(['eval $odoth' 10]));
mssendraw(sckCam,uint8(['imageget all' 10]));

if nargin >2
    mssendraw(sckLaser,uint8(['scanget codex=TAG' 10]));
end

pause(0.2);

[resp, flag,headerEnd]=receiveImage(sckCam);

if flag==0
    display('failed to get image');
    image=0;
    pose=0;
    absPose=0;
    scanData=[];
    return
end

B=resp(headerEnd+1:3:end);
G=resp(headerEnd+2:3:end);
R=resp(headerEnd+3:3:end);

height=length(B)/320;

I=zeros(height,320,3);

I(1:height,:,1)=reshape(R,320,height,1)';
I(1:height,:,2)=reshape(G,320,height,1)';
I(1:height,:,3)=reshape(B,320,height,1)';

image=I/255;

response=char(receiveMessage(sckDemo,0.1,0.02));
pose=sscanf(response,'%f')';

if length(pose)<3
    display('failed to get pose');
    image=0;
    pose=0;
    absPose=0;
    scanData=[];
    return
end

absPose=pose;

pose=evalPose(pose);

scanData=[];

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