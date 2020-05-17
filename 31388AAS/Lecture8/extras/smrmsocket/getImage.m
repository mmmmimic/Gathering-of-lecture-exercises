function [ I ] = getImage( socket )

I=[];

receiveMessage(socket,0.1,0.1,1000000);
mssendraw(socket,uint8(['imageget all' 10]));

[resp, flag,headerEnd]=receiveImage(socket);

if flag==0
    display('failed to get image');
end

B=resp(headerEnd+1:3:end);
G=resp(headerEnd+2:3:end);
R=resp(headerEnd+3:3:end);

height=length(B)/320;

I=zeros(height,320,3);

I(1:height,:,1)=reshape(R,320,height,1)';
I(1:height,:,2)=reshape(G,320,height,1)';
I(1:height,:,3)=reshape(B,320,height,1)';

I=I/255;