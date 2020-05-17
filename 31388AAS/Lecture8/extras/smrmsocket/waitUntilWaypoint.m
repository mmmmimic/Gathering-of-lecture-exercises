function waitUntilWaypoint(mrcSck)
response = '';
while(true)
    response = [response char(receiveMessage(mrcSck,0.01))];
    if(~isempty(strfind(response,'syncevent complete')))
        break
    end
end

success = 1;