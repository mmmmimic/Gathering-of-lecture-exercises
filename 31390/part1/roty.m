function [rotmtx] = roty(angle)
% return the rotation matrix around x axis
% angle: degree
rotmtx = [cosd(angle), 0, sind(angle); 0, 1, 0; -sind(angle), 0, cosd(angle)];
end

