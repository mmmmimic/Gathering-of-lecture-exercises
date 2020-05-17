function [ polarLines ] = convertToPolar( endpointsLines )

    noOfLines = size(endpointsLines,2);
    polarLines = zeros(2,noOfLines);
    for i=1:noOfLines
        line = endpointsLines(:,i);
        polarLines(:,i) = calculateLine([line(1:2), line(3:4)]);
    end

end
