function [ poseNew, covNew ] = addPoseProbabilistic( poseOld, covOld, poseDisp, covDisp)
%ADDPOSEPROBABILISTIC Summary of this function goes here
%   Detailed explanation goes here

poseNew = addPose(poseOld,poseDisp);

th = poseOld(3);
xdx = poseDisp(1);
xdy = poseDisp(2);

transMat = [1 0 -sin(th)*xdx-cos(th)*xdy cos(th) -sin(th) 0
            0 1 cos(th)*xdx-sin(th)*xdy  sin(th) cos(th)  0
            0 0 1                        0        0       1];
        
covNew = transMat*[covOld,zeros(3,3);zeros(3,3),covDisp]*transMat';

end
