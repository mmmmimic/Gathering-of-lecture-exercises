function R = v2w(v,w)
% return the rotation matrix R from v to w
r = cross(v,w);
rx = r(1);
ry = r(2);
rz = r(3);
alpha = atan2(ry, rx);
beta = atan2(sqrt(rx^2+ry^2), rz);
theta = acos(v'*w);
alpha = alpha/pi*180;
beta = beta/pi*180;
theta = theta/pi*180;
R = rotz(alpha)*roty(beta)*rotz(theta)*roty(-beta)*rotz(-alpha);
end

