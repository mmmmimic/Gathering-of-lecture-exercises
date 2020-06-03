function [ message, flag,headerEnd ] = receiveImage( socket )
flag=1;

submessage=zeros(960,240);

header=msrecvraw(socket,300,1)';
headerEnd=0;
if length(header)~=300
    flag=0;
    message=[];
    return
end

In=find(header==10);

if length(In)<1
    flag=0;
    message=[];
    return
else
    res=sscanf(char(header(In(1)+1:end)),'<bin size="%d" codex=%s');
    if length(res)<6 || all(char(res(2:6)')~='"BIN"')
        flag=0;
    end
    headerEnd=find(char(header)=='>');
    if length(headerEnd)<2
        flag=0;
        message=[];
        return
    end    
    headerEnd=headerEnd(2);
end

correction=msrecvraw(socket,960-length(header)+headerEnd,0.1)';

k=1;
while 1
    tmp=msrecvraw(socket,960,0.1);
    if length(tmp)~=0
        submessage(:,k)=tmp;
    else
        break;
    end
    k=k+1;
end

submessage=submessage(:,1:k-1);

message=[header correction submessage(:)'];