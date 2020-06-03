function [ lR lL ] = odoFindLrLl( p1, p2, Bin )
%ODOFINDLRLL(p1,p2,B) Find the left and right wheel displacements from two
%consecutive odometry poses
%   p1, p2 are the first and the second pose. Bin is the wheel separation

D=mod(p2(3)-p1(3)+3*pi,2*pi)-pi;
L=sqrt((p2(1)-p1(1))^2+(p2(2)-p1(2))^2);
lR=L+D*Bin/2;
lL=2*L-lR;

end
