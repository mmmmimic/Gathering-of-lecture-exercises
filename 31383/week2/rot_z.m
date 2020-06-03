function rotmat = rot_z(gamma)
rotmat = [cos(gamma) -sin(gamma) 0 0; sin(gamma) cos(gamma) 0 0; 0 0 1 0;0 0 0 1];
end