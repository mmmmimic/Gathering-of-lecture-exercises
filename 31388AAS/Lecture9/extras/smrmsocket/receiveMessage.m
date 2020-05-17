function [ message ] = receiveMessage( socket, waitTime, waitTimeShort, msgLength )
% [ message ] = receiveMessage( socket, waitTime, waitTimeShort )
% A convenience function to receive a message coming through a socket
% connection. It returns the entire available message. Note that the
% available message can also include responses to previous commands, in
% order to remove whatever is available call this function once and then
% issue the command.
%
% socket: the handle object for the socket connection
% waitTime: The initial duration for timeout, returns empty string if no
%   data is available initially for this period of time. Default = 1
% waitTimeShort: The maximum amount of time to wait for consecutive
%   characters once the first character is received. Default = waitTime
% msgLength: The length of each sub-message. Giving a value of 1 here will
%   return any message correctly, but for long messages grabing the message
%   one byte at a time can take too much time, larger values here will
%   increase the speed, but my clip the message at the end. Default = 1
%
% message: The received message, it is a row uint8 array


message=[];

if nargin<4
    msgLength=1;
end

bufSizeB=floor(1000/msgLength);
bufSize=bufSizeB*msgLength;

submessage=zeros(1,bufSize);
k=1;

if nargin==1
        tmp=msrecvraw(socket,msgLength,1)';
else
        tmp=msrecvraw(socket,msgLength,waitTime)';
end        

if nargin==2
    waitTimeShort=waitTime;
end

while 1    
    if length(tmp)~=0
        submessage((k-1)*msgLength+1:k*msgLength)=tmp;
    else
        break;
    end
    
    if k==bufSizeB
    message=[message submessage]; %#ok<AGROW>
    k=0;
    end
    k=k+1;
    if length(message)>1000000
        break;
    end
    
    if nargin==1
        tmp=msrecvraw(socket,msgLength,1)';
    else
        tmp=msrecvraw(socket,msgLength,waitTimeShort)';
    end        
end
message=[message submessage(1:(k-1)*msgLength)];