function rotmat=rot_x(alpha)
rotmat = [1 0 0 0;0 cos(alpha) -sin(alpha) 0; 0 sin(alpha) cos(alpha) 0;0 0 0 1];
end