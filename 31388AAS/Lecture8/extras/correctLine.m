function [ line ] = correctLine( line )
%CORRECTLINE Summary of this function goes here
%   Detailed explanation goes here
    theta = line(1);
    radius = line(2);
    if(radius<0) 
		radius=-radius;
        if(theta<=0) 
            theta=theta+pi;
        else
            theta=theta-pi;
        end
    end

    theta = mod(theta+pi,2*pi)-pi;
    
    line = [theta;radius];

end
