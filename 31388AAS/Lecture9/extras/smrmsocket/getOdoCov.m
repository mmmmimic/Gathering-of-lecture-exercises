function [ odoCov ] = getOdoCov( socket)

covElements=[];
receiveMessage(socket,0.02);
mssendraw(socket,uint8(['eval $odocovx' 10]));
mssendraw(socket,uint8(['eval $odocovxy' 10]));
mssendraw(socket,uint8(['eval $odocovthx' 10]));
mssendraw(socket,uint8(['eval $odocovy' 10]));
mssendraw(socket,uint8(['eval $odocovthy' 10]));
mssendraw(socket,uint8(['eval $odocovth' 10]));
while(length(covElements)<6)
    response=char(receiveMessage(socket,0.1,0.1));
    covElements=[covElements sscanf(response,'%f')']; %#ok<AGROW>
end

odoCov = [covElements(1) covElements(2) covElements(3)
          covElements(2) covElements(4) covElements(5)
          covElements(3) covElements(5) covElements(6)];