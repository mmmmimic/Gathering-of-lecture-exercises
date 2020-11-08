t = [-1.5; -0.5; 0.5; 1.5; 2.5];
y = [0.80; 1.23; 1.15; 1.48; 2.17];

A = [t, ones(size(t))];


%% contour plot


x1 = 0:0.01:0.5;
x2 = 1:0.01:1.5;
[X,Y] = meshgrid(x1,x2);

F2 = 0;
F1 = 0;
tmp = [];
for ii = 1:length(t)
    F2 = F2+(t(ii)*X+Y-y(ii)).^2;
    F1 = F1+abs(t(ii)*X+Y-y(ii));
    tmp(:,:,ii) = abs(t(ii)*X+Y-y(ii)); 
end
F2 = sqrt(F2);
Finf = max(tmp,[],3);

k = 0:9;
v1 = 0.78+0.15*k;
v2 = 0.4+0.1*k;
vinf = 0.22+0.1*k;

figure,
subplot(1,3,1),
[c,h]=contour(X,Y,F2,v2,'linewidth',2); 
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
title('l_2')

subplot(1,3,2),
[c,h]=contour(X,Y,F1,v1,'linewidth',2); 
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
title('l_1')

subplot(1,3,3),
[c,h]=contour(X,Y,Finf,vinf,'linewidth',2); 
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
title('l_\infty')

