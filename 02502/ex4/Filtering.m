close all;clear all; clc;
%%
%ex1
f = [zeros(5,3),ones(5,2)];
h = ones(3,3);
g = imfilter(f,h)

%%
%ex2
im1 = imread('Gaussian.png');
imshow(im1);
% create mean filter
fsize = 5;
h = ones(fsize)/fsize^2;

%%
%ex3
meanim1 = imfilter(im1,h);
figure;
subplot(1,2,1);
imshow(im1), colormap(gca,gray), axis image off;
title('Original image');
subplot(1,2,2);
imshow(meanim1), colormap(gca,gray), axis image off;
title('Filtered image, mean filter');
%%
figure;
subplot(1,2,1);
imshow(im1), colormap(gca,hot), axis image off;
title('Original image');
subplot(1,2,2);
imshow(meanim1), colormap(gca,hot), axis image off;
title('Filtered image, mean filter');

%%
%ex4
meanim2 = imfilter(im1,h,'replicate');
figure;
subplot(1,2,1);
imshow(meanim1), colormap(gca,hot), axis image off;
title('Filtered image, mean filter 1');
subplot(1,2,2);
imshow(meanim2), colormap(gca,hot), axis image off;
title('Filtered image, mean filter 2');

%%
%ex5
% Yes yes

%%
%ex6
fsize = 15;
h = ones(fsize)/fsize^2;
meanim1 = imfilter(im1,h);
figure;
subplot(1,2,1);
imshow(im1), colormap(gca,hot), axis image off;
title('Original image');
subplot(1,2,2);
imshow(meanim1), colormap(gca,hot), axis image off;
title('Filtered image, mean filter');
% blurring

%%
%ex7
% They are extended

%%
%ex8
medim1 = medfilt2(im1,[5,5]);
figure;
subplot(1,2,1);
imshow(meanim1), colormap(gca,hot), axis image off;
title('Mean filter');
subplot(1,2,2);
imshow(medim1), colormap(gca,hot), axis image off;
title('Median filter');

%%
%ex9
medim2 = medfilt2(im1,[15,15]);
figure;
subplot(1,2,1);
imshow(meanim2), colormap(gca,hot), axis image off;
title('Mean filter');
subplot(1,2,2);
imshow(medim2), colormap(gca,hot), axis image off;
title('Median filter');

%%
%ex10
% mean filter: size->blur
% median filter: size-/> blur

%%
%ex11
salt = imread('SaltPepper.png');
fsize = 5;
h = ones(fsize)/fsize^2;
meanim3 = imfilter(salt,h);
medim3 = medfilt2(salt,[5,5]);
figure;
subplot(1,3,1);
imshow(salt), colormap(gca,gray), axis image off;
title('Origional picture');
subplot(1,3,2);
imshow(meanim3), colormap(gca,gray), axis image off;
title('Median filter');
subplot(1,3,3);
imshow(medim3), colormap(gca,gray), axis image off;
title('Median filter');

%%
%ex12
H = fspecial('average',[5,5]);
H-h

%%
%ex13
CT = imread('ElbowCTSlice.png');
sobel = fspecial('sobel');
CT1 = imfilter(CT,sobel);
figure;
subplot(1,2,1);
imshow(CT,[]), colormap(gca,hot), axis image off;
title('Origional picture');
subplot(1,2,2);
imshow(CT1,[]), colormap(gca,hot), axis image off;
title('Sobel filter');
%%
%ex14
sobel_t = sobel';
sobel_mt = -sobel';
CT2_1 = imfilter(CT,sobel_t);
CT2_2 = imfilter(CT,sobel_mt);
figure;
subplot(1,3,1);
imshow(CT,[]), colormap(gca,hot), axis image off;
title('Origional picture');
subplot(1,3,2);
imshow(CT2_1,[]), colormap(gca,hot), axis image off;
title('Sobel filter 90');
subplot(1,3,3);
imshow(CT2_2,[]), colormap(gca,hot), axis image off;
title('Sobel filter -90');

%%
%ex15
CT1_1 = find_edge(CT,0);
CT1_2 = find_edge(CT,1);
h = fspecial('average',[5,5]);
CT3 = imfilter(CT,h);
CT3_1 = find_edge(CT3,0);
CT3_2 = find_edge(CT3,1);
h = fspecial('average',[13,13]);
CT4 = imfilter(CT,h);
CT4_1 = find_edge(CT4,0);
CT4_2 = find_edge(CT4,1);
figure;
subplot(3,3,1);
imshow(CT,[]), colormap(gca,hot), axis image off;
title('Origional picture');
subplot(3,3,2);
imshow(CT1_1,[]), colormap(gca,hot), axis image off;
title('Prewitt filter');
subplot(3,3,3);
imshow(CT2_1,[]), colormap(gca,hot), axis image off;
title('Sobel filter');
subplot(3,3,4);
imshow(CT3,[]), colormap(gca,hot), axis image off;
title('Mean Filter [5,5]');
subplot(3,3,5);
imshow(CT3_1,[]), colormap(gca,hot), axis image off;
title('Prewitt filter');
subplot(3,3,6);
imshow(CT3_2,[]), colormap(gca,hot), axis image off;
title('Sobel filter Vertical');
subplot(3,3,7);
imshow(CT4,[]), colormap(gca,hot), axis image off;
title('Mean Filter [13,13]');
subplot(3,3,8);
imshow(CT4_1,[]), colormap(gca,hot), axis image off;
title('Priwitt filter');
subplot(3,3,9);
imshow(CT4_2,[]), colormap(gca,hot), axis image off;
title('Sober filter');

%%
%ex16
CT5 = medfilt2(CT,[5,5]);
CT5_1 = find_edge(CT5,0);
CT5_2 = find_edge(CT5,1);
CT6 = medfilt2(CT,[13,13]);
CT6_1 = find_edge(CT6,0);
CT6_2 = find_edge(CT6,1);
figure;
subplot(3,3,1);
imshow(CT,[]), colormap(gca,hot), axis image off;
title('Origional picture');
subplot(3,3,2);
imshow(CT1_1,[]), colormap(gca,hot), axis image off;
title('Prewitt filter');
subplot(3,3,3);
imshow(CT2_1,[]), colormap(gca,hot), axis image off;
title('Sobel filter');
subplot(3,3,4);
imshow(CT3,[]), colormap(gca,hot), axis image off;
title('Median Filter [5,5]');
subplot(3,3,5);
imshow(CT5_1,[]), colormap(gca,hot), axis image off;
title('Prewitt filter');
subplot(3,3,6);
imshow(CT5_2,[]), colormap(gca,hot), axis image off;
title('Sobel filter Vertical');
subplot(3,3,7);
imshow(CT6,[]), colormap(gca,hot), axis image off;
title('Median Filter [13,13]');
subplot(3,3,8);
imshow(CT6_1,[]), colormap(gca,hot), axis image off;
title('Priwitt filter');
subplot(3,3,9);
imshow(CT6_2,[]), colormap(gca,hot), axis image off;
title('Sober filter');
% better

%%
%ex17
CT_edge = edge(CT);
imshow(CT_edge),colormap(gca,pink),axis image off;

%%
%ex18
G = fspecial('Gaussian',17,3);
surf(G);

%%
%ex19
sigma = [1,3,11];
figure;
for i = 1:3
   subplot(1,3,i);
   filter = fspecial('Gaussian',51,sigma(i));
   temp = imfilter(CT,filter);
   imshow(temp),colormap(gca,gray),axis image off;
end

%%
%ex20
book = imread('book.jpg');
imshow(book);
book = rgb2gray(book);
book = imresize(book,[1000,NaN]);
imshow(book);

%%
%ex21
filter = fspecial('Gaussian',5,11);
book1 = find_edge(book,0);
book2 = find_edge(book,1);
book3 = edge(book);
book4 = imfilter(book,filter);
figure;
subplot(2,2,1)
imshow(book1,[]), colormap(gca,hot), axis image off;
title('Prewitt filter');
subplot(2,2,2)
imshow(book2,[]), colormap(gca,hot), axis image off;
title('Sobel filter');
subplot(2,2,3)
imshow(book3,[]), colormap(gca,hot), axis image off;
title('Edge');
subplot(2,2,4)
imshow(book4,[]), colormap(gca,gray), axis image off;
title('Gaussian filter');
