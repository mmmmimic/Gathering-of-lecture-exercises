function [rotmtx] = rotz(angle)
% return the rotation matrix around x axis
% angle: degree
rotmtx = [cosd(angle), -sind(angle), 0; sind(angle), cosd(angle), 0; 0, 0, 1];
end
