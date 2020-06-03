function [hor ver] = CameraFOV(len, wid, b)
% hor: horizonal FOV
% ver: vertical FOV
% len: lenth of CCD
% wid: width of CCD
% b: CCD distance to camera
hor = deg2rag(2*atan(wid/(2*b)));
ver = deg2rag(2*atan(len/(2*b)));
end