%%
clear all;close all;clc;
%%
ct2 = dicomread('CTangio2.dcm');
I2 = imread('CTAngio2Scaled.png');
imshow(I2);
%%
% ex1
% backgground
BGROI = roipoly(I2);
imwrite(BGROI, 'BGROI.png');
BGVals = double(ct2(BGROI));
% fat
FatROI = roipoly(I2);
imwrite(FatROI, 'FatROI.png');
FatVals = double(ct2(FatROI));
% liver
LiverROI = roipoly(I2);
imwrite(LiverROI, 'LiverROI.png');
LiverVals = double(ct2(LiverROI));
% kidney
KidROI = roipoly(I2);
imwrite(KidROI, 'KidROI.png');
KidVals = double(ct2(KidROI));
% spleen
SpROI = roipoly(I2);
imwrite(SpROI, 'SpROI.png');
SpVals = double(ct2(SpROI));
% trabecular bone
TrROI = roipoly(I2);
imwrite(TrROI, 'TrROI.png');
TrVals = double(ct2(TrROI));
% hard bone
HdROI = roipoly(I2);
imwrite(HdROI, 'HdROI.png');
HdVals = double(ct2(HdROI));

%%
% ex2
% background
figure;
hist(BGVals);
sprintf('Background mean %g std %g min %g max %d',...
mean(BGVals), std(BGVals), ...
min(BGVals), max(BGVals))
% fat
figure;
hist(FatVals);
sprintf('Fat mean %g std %g min %g max %d',...
mean(FatVals), std(FatVals), ...
min(FatVals), max(FatVals))
%liver
figure;
hist(LiverVals);
sprintf('Liver mean %g std %g min %g max %d',...
mean(LiverVals), std(LiverVals), ...
min(LiverVals), max(LiverVals))
% kidney
figure;
hist(KidVals);
sprintf('Kidney mean %g std %g min %g max %d',...
mean(KidVals), std(KidVals), ...
min(KidVals), max(KidVals))
% spleen
% Gaussian Distribution
figure;
hist(SpVals);
sprintf('Spleen mean %g std %g min %g max %d',...
mean(SpVals), std(SpVals), ...
min(SpVals), max(SpVals))
% trabecular bone
figure;
hist(TrVals);
sprintf('Trabecular bone mean %g std %g min %g max %d',...
mean(TrVals), std(TrVals), ...
min(TrVals), max(TrVals))
% hard bone
figure;
hist(HdVals);
sprintf('Hard bone mean %g std %g min %g max %d',...
mean(HdVals), std(HdVals), ...
min(HdVals), max(HdVals))

%%
% ex3
BGfit = pllot(BGVals);
Fatfit = pllot(FatVals);
Liverfit = pllot(LiverVals);
Kidfit = pllot(KidVals);
Spfit = pllot(SpVals);
Trfit = pllot(TrVals);
Hdfit = pllot(HdVals);

%%
% ex4
xrange = -1200:0.1:1200;
plot(xrange,BGfit, xrange, Trfit, xrange, ...
Liverfit, xrange, Spfit, xrange,Kidfit, ...
xrange, Fatfit, xrange, Hdfit);
legend('Background','Trabeculae', 'Liver', 'Spleen',...
'Kidney', 'Fat','Bone');
xlim([-1200 1200]);

% collapse liver and spleen into one class

%%
% ex5
var = BGVals;
mean_v_1 = mean(var);
var = KidVals;
mean_v_2 = mean(var);
T1 = (mean_v_1+mean_v_2)/2;
var = FatVals;
mean_v_1 = mean(var);
T2 = (mean_v_1+mean_v_2)/2;
var = LiverVals;
mean_v_2 = mean(var);
T3 = (mean_v_1+mean_v_2)/2;
var = TrVals;
mean_v_1 = mean(var);
T4 = (mean_v_1+mean_v_2)/2;
var = HdVals;
mean_v_2 = mean(var);
T5 = (mean_v_1+mean_v_2)/2;
%%
% ex6
ILabel = LabelImage(ct2, T1, T2, T3, T4, T5);
imagesc(ILabel)

%%
% ex7
figure
imagesc(ILabel)
hcb=colorbar;
set(hcb,'YTick',[0,1,2,3,4,5]);
set(hcb,'YTickLabel',{'Class 0', 'Class 1','Class 2',...
'Class 3','Class 4','Class 5'});
%%
% ex8
T1 = -74.1;
T2 = 12.8;
T3 = 34.6;
T4 = 77.4;
T5 = 322.2;

%%
% ex9
ILabel2 = LabelImage(ct2, T1, T2, T3, T4, T5);
imagesc(ILabel2)


%%
% ex10
I = imread('DTUSigns055.jpg');
Ired = I(:,:,1);
Igreen = I(:,:,2);
Iblue = I(:,:,3);
% select the rigion
% red sign
RROI = roipoly(I);
imwrite(RROI, 'RROI.png');
redVals = double(Ired(RROI));
greenVals = double(Igreen(RROI));
blueVals = double(Iblue(RROI));
figure;
RtotVals = [redVals greenVals blueVals];
% blue sign
BROI = roipoly(I);
imwrite(BROI, 'BROI.png');
redVals = double(Ired(BROI));
greenVals = double(Igreen(BROI));
blueVals = double(Iblue(BROI));
figure;
BtotVals = [redVals greenVals blueVals];
% white car
WROI = roipoly(I);
imwrite(WROI, 'WROI.png');
redVals = double(Ired(WROI));
greenVals = double(Igreen(WROI));
blueVals = double(Iblue(WROI));
figure;
WtotVals = [redVals greenVals blueVals];
% green leaves
GROI = roipoly(I);
imwrite(GROI, 'GROI.png');
redVals = double(Ired(GROI));
greenVals = double(Igreen(GROI));
blueVals = double(Iblue(GROI));
figure;
GtotVals = [redVals greenVals blueVals];
% yellow glass
YROI = roipoly(I);
imwrite(YROI, 'YROI.png');
redVals = double(Ired(YROI));
greenVals = double(Igreen(YROI));
blueVals = double(Iblue(YROI));
%%
figure;
totVals = RtotVals;
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);
%%
figure;
totVals = BtotVals;
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);
%%
figure;
totVals = WtotVals;
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);
%%
figure;
totVals = GtotVals;
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);
%%
figure;
totVals = YtotVals;
nbins = 255;
hist(totVals,nbins);
h = findobj(gca,'Type','patch');
set(h(3),'FaceColor','r','EdgeColor','r','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(2),'FaceColor','g','EdgeColor','g','FaceAlpha',0.3,'EdgeAlpha',0.3);
set(h(1),'FaceColor','b','EdgeColor','b','FaceAlpha',0.3,'EdgeAlpha',0.3);
xlim([0 255]);
%%
% ex11 12 13 14
RR = [157 175];
RG = [47.5,67.3];
RB = [49.8 65.8];
figure;
pic = Ired > RR(1) & Ired < RR(2) & Igreen > RG(1) & Igreen < RG(2) & Iblue > RB(1) & Iblue < RB(2);
imshow(pic);
%%
RR = [0 12.1];
RG = [85.8,102];
RB = [183 197];
figure;
pic = Ired > RR(1) & Ired < RR(2) & Igreen > RG(1) & Igreen < RG(2) & Iblue > RB(1) & Iblue < RB(2);
imshow(pic);
%%
RR = [107 123];
RG = [114,130];
RB = [120 133];
figure;
pic = Ired > RR(1) & Ired < RR(2) & Igreen > RG(1) & Igreen < RG(2) & Iblue > RB(1) & Iblue < RB(2);
imshow(pic);
%%
RR = [0 193];
RG = [0,193];
RB = [0 134];
figure;
pic = Ired > RR(1) & Ired < RR(2) & Igreen > RG(1) & Igreen < RG(2) & Iblue > RB(1) & Iblue < RB(2);
imshow(pic);
%%
RR = [108 121];
RG = [113,131];
RB = [119 135];
figure;
pic = Ired > RR(1) & Ired < RR(2) & Igreen > RG(1) & Igreen < RG(2) & Iblue > RB(1) & Iblue < RB(2);
imshow(pic);
%%
% function definition
function pdfFit = pllot(Var) 
figure;
xrange = -1200:0.1:1200; % Fit over the complete Hounsfield range
pdfFit = normpdf(xrange, mean(Var), std(Var));
S = length(Var); % A simple scale factor
hold on;
hist(Var,xrange);
plot(xrange, pdfFit * S,'r');
hold off;
xlim([-10, 100]);
end