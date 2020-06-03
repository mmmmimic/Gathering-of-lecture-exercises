close all;
clear all;
clc;
%% ex1
mc = imread('metacarpals.png');
mc_size = size(mc);
whos
%% ex2
imshow(mc);
imhist(mc);
%% ex3 
[counts, x] = imhist(mc);
find(counts == max(counts))

%% ex4
mc(100,90)

%% ex5
imtool(mc)

%% ex6
ho = imread('horns.jpg');
ho_1 = imresize(ho,0.25);
whos
subplot(2,1,1);
imshow(ho_1);
impixel(ho_1, 500, 400)
ho_gry = rgb2gray(ho_1);
subplot(2,1,2);
imshow(ho_gry);
imhist(ho_1);
sea = imread('sea.jpg');
to = imread('tower.jpg');
sea_1 = imresize(sea, 0.25);
to_1 = imresize(to, 0.25);
subplot(2,1,1)
imshow(sea_1);
subplot(2,1,2)
imshow(to_1);
figure;
subplot(2,1,1);
imhist(sea_1);
subplot(2,1,2);
imhist(to_1);


% ct
ctInf = dicominfo('CTangio.dcm');
ctInf
ct = dicomread('CTangio.dcm');
whos;
imtool(ct);


%
mc = imread('metacarpals.png');
imshow(mc);
colormap(gca, cool);

%
im1 = imread('DTUSign1.jpg');
imshow(im1);
Rcomp = im1(:,:,1);
figure;
subplot(3,1,1);
imshow(Rcomp);
colormap(gca, gray);
Gcomp = im1(:,:,2);
subplot(3,1,2);
imshow(Gcomp);
colormap(gca, gray);
Bcomp = im1(:,:,3);
subplot(3,1,3);
imshow(Bcomp);
colormap(gca, gray);
figure;
subplot(3,1,1);
imhist(Rcomp);
subplot(3,1,2);
imhist(Gcomp);
subplot(3,1,3);
imhist(Bcomp);
im1(500:1000,800:1500,:)=0;
imwrite(im1,'DTUSign1-marked.jpg');
im1(1567:1722,2304:2748,:) = 0;
imwrite(im1,'DTUSign1-marked.png');
imtool(im1);

%%
fing = imread('finger.png');
imcontour(fing, 5);
figure;
imshow(fing);
improfile;
mesh(double(fing));