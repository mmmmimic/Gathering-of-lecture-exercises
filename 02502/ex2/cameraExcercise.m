clear all;
%% question 1
% calculate the angle
a = 10;
b = 3;
theta = atan(b/a);


%% question 2
f = 15;
g1 = 100;
g2 = 1000;
g3 = 5000;
g4 = 15000;
b1 = cameraBDistance(f,g1)
b2 = cameraBDistance(f,g2)
b3 = cameraBDistance(f,g3)
b4 = cameraBDistance(f,g4)

%% question 3
info = imfinfo('1.jpg');
info.DigitalCamera;

%% question 4
% 1)
f = 5;
g = 5000;
b = cameraBDistance(f,g);

% 2)
h = 1800;
t = RealSizeOnCCD(b,g,h);

% 3)
len = 6.4;
wid = 4.8;
p_l_num = 640;
p_w_num = 480;
pixel_len = len/p_l_num;
pixel_wid = wid/p_w_num;
pixel_size = string(pixel_len)+'*'+string(pixel_wid);

% 4)
p_t = PixelSizeOnCCD(h, g);

% 5) 6)
[hor, ver] = CameraFOV(len,wid,b);

%%
%

