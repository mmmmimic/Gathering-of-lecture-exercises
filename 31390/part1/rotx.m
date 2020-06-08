function [rotmtx] = rotx(angle)
% return the rotation matrix around x axis
% angle: degree
rotmtx = [1, 0, 0; 0, cosd(angle), -sind(angle); 0, sind(angle), cosd(angle)];
end

