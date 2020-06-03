clear all;close all; clc;
vb=imread("vertebra.png");
imshow(vb);
figure;
imhist(vb);
max_pix = max(max(vb));
min_pix = min(min(vb));
%%
%e1
mean_pix = mean2(vb);
%%
% e2
Io = HistStretch(vb);
imhist(Io);
%%
% e3
figure;
gamma = [0.48, 1, 1.52];
x = [0:0.01:1];
for i = 1:3
    plot(x,x.^(gamma(i)));
    hold on;
end

%%
%e4
figure;
for i = 1:3
    subplot(1,3,i)
    To_temp = gamma_mapping(Io,gamma(i));
    imshow(To_temp);
end

%%
%e5
lowout = 0; highout = 1;
Img = imadjust(Io, [], [lowout highout]);
figure;
subplot(2,1,1);
imshow(Io);
subplot(2,1,2);
imshow(Img);

%%
%e6
Img2 = ImgThreshold(Io, 0.3);
figure;
imagesc(Img2);
colormap(gray);

%%
%e7
th = graythresh(Io);
Img3 = ImgThreshold(Io, th);
figure;
imagesc(Img3);
colormap(gray);

%%
%e8
img = imread("dark_background.png");
figure;
subplot(3,1,1)
imshow(img);
img = rgb2gray(img);
subplot(3,1,2)
imshow(img);
th = graythresh(img);
Img = ImgThreshold(img,th);
subplot(3,1,3)
imagesc(Img);
colormap(gray);

%%
%e9
imtool(vb);

%%
%e10
im = imread('DTUSigns2.jpg');
imshow(im);
Rcomp = im(:,:,1);
Gcomp = im(:,:,2);
Bcomp = im(:,:,3);
segm = Rcomp < 10 & Gcomp > 85 & Gcomp < 105 & Bcomp > 180 & Bcomp < 200;
seg = im(1466:1619,2034:2400,:);% extract the board part
figure;
for i = 1:3
    subplot(3,1,i)
    imhist(seg(:,:,i))
end
segm1 = Rcomp<178&Rcomp>150&Gcomp>42&Gcomp<67&Bcomp>41&Bcomp<70;
figure;
imshow(segm1);


%%
%e11
HSV = rgb2hsv(im);
Hcomp = HSV(:,:,1);
Scomp = HSV(:,:,2);
Vcomp = HSV(:,:,3);
figure;
imshow(HSV);
colormap(hsv);
figure;
subplot(3,1,1);
colormap(gray);
imshow(Hcomp);
subplot(3,1,2);
imshow(Scomp);
colormap(gray);
subplot(3,1,3);
imshow(Vcomp);
colormap(gray);
%
cor_arrow = [1549,2101;582,1068];
cor_board = [1466,1619;2034,2400];
arrow = HSV(cor_arrow(1,1):cor_arrow(1,2),cor_arrow(2,1):cor_arrow(2,2),:);
imshow(arrow);
colormap(hsv);
figure;
for i=1:3
    subplot(3,1,i)
    imhist(arrow(:,:,i));
end
arrow_segm = Hcomp<0.6&Hcomp>0.54&Scomp>0.95&Scomp<1&Vcomp>0.7&Vcomp<0.8;
figure;
imshow(arrow_segm);
%
board = HSV(cor_board(1,1):cor_board(1,2),cor_board(2,1):cor_board(2,2),:);
imshow(board);
colormap(hsv);
figure;
for i=1:3
    subplot(3,1,i)
    imhist(board(:,:,i));
end
figure;
board_segm = Hcomp<1&Hcomp>0.97&Scomp>0.6&Scomp<0.7&Vcomp>0.6&Vcomp<0.7;
imshow(board_segm);