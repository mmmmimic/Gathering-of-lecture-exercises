function [lines,lCoords] = SampleSpokes(IM, nAng, length, center)

lCoords= zeros(nAng,length,2);
lines = zeros(nAng,length); % pixel values along wheel spokes

startAng = 0; % starting angle (to get a nice plot - try e.g. with 0) 

% sample spokes
for I=1:nAng
    ang = 2*pi*I/nAng + startAng; % spoke angle 
    t = [cos(ang) sin(ang)]; % unit vector from angle
 %   plot([center(1) t(1) * maxL + center(1)], [center(2) t(2) * maxL  + center(2)]);
    
    % compute coordinates of pixels along spokes using Nearest Neighboorhood interpolation
    linecoords = ones(length,1)*center + (0:length-1)'*t; % get coordinates for interpolation
    % store coordinates as pixel index in original image
    lCoords(I,:,:) = linecoords;  
    % extract pixel values using bilinear interpolation
    lines(I,:) = interp2(IM,linecoords(:,1),linecoords(:,2),'linear',0)'; 
end
