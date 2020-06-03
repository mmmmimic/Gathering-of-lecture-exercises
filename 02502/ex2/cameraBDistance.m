function b = CameraBDistance(f,g)
% b: the distance where the CCD should be placed
% f: focal lenth
% g: object distance
b = 1/(1/f-1/g);
end