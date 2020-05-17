function [ Lines, ImWidth, ImHeight ] = receiveLines( sckCam )

receiveImage(sckCam);
receiveMessage(sckCam,0.02,0.02,100);

mssendraw(sckCam,uint8(['hough device=10 parametric' 10]));

header=char(msrecvraw(sckCam,57,1))';

if length(header)>10 && strcmp(header(1:10),'NoOfLines:')==1
   NoOfLines = sscanf(header(1:15),'NoOfLines: %d');
   BytesPerLine = sscanf(header(16:32), ' BytesPerLine: %d');
   ImWidth = sscanf(header(34:44),'Width: %d');
   ImHeight = sscanf(header(46:end), 'Height: %d');
else
    Lines=[];
    ImWidth=[];
    ImHeight=[];
    display(['Couldnt resolve header: ' header]);
    return;
end

messageSize=NoOfLines*BytesPerLine;

rest=char(msrecvraw(sckCam,messageSize,0.1)');

lineData = sscanf(rest, ' %f %f %f %f');

Lines=reshape(lineData,4, length(lineData)/4)';
