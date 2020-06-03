function [ matchResult ] = match( pose, poseCov, worldLines, laserLines )
% [matchResult] = MATCH(pose,poseCov,worldLines,laserLines)
%   This function matches the predicted lines to the extracted lines. The
%   criterion for a match is the mahalanobis distance between the (alpha,
%   r) parameters of the predicted and the extracted lines. The arguments
%   are:
%       pose: The estimated robot pose given as [x,y,theta]
%       poseCov: The estimated covariance matrix of the robot pose
%       worldLines: Known world lines in world coordinates, given as
%       [alpha;r] for each line. Number of columns = number of lines
%       laserLines: Lines extracted from the laser scan. Given as [alpha;r]
%       for each line. Number of columns = number of lines
%
%       matchResult: A (5xnoOfWorldLines) matrix whose columns are
%       individual pairs of line matches. It is structured as follows:
%       matchResult = [ worldLine(1,1) , worldLine(1,2) ...  ]
%                     [ worldLine(2,1) , worldLine(2,2)      ]
%                     [ innovation1(1) , innovation2(1)      ]
%                     [ innovation1(2) , innovation2(2)      ]
%                     [ matchIndex1    , matchIndex2    ...  ]
%           Note that the worldLines are in the world coordinates!

% The varAlpha and varR are the assumed variances of the parameters of
% the extracted lines, they are read globally.
global varAlpha varR
noOfWorldLines = size(worldLines, 2);
noOfLaserLines = size(laserLines, 2);
matchResult = zeros(5, noOfWorldLines);
g = 2;
for i = 1:noOfWorldLines
    worldLine = worldLines(:,i);
    [projLine, worldLineCov] = projectToLaser(worldLine,pose, poseCov);
    % There is only 1 covariance matrix for each world line
    C = worldLineCov+diag([varAlpha,varR]);
    dist = [];
    idx = [];
    innovation = [];
    for j = 1:noOfLaserLines
        laserLine = laserLines(:,j);
        v = laserLine - projLine;
        % Mahalanobis distance
        dist_m = v'*inv(C)*v;
        if dist_m<=g^2
            dist = [dist, dist_m];
            idx = [idx, j];
            innovation = [innovation, v];
        end
    end
    % find the best match
    if ~isempty(dist) % dist is not []
        [~, ind] = min(dist);
        idx = idx(ind);
        innovation = innovation(:,ind);
    else
        idx = 0;
        innovation = [0;0];
    end
    matchResult(:, i) = [worldLine;innovation;idx];
end

end
