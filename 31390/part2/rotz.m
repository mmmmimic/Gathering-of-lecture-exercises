function [rotmtx] = rotz(angle)
% return the rotation matrix around x axis
% angle: degree
rotmtx = [cos(angle), -sin(angle), 0; sin(angle), cos(angle), 0; 0, 0, 1];
end
