%%
clear; close all; clc;

%% Question 3
load vec_b.mat

n = length(b);
[A, L] = get_AL(n);

figure, 
subplot(1,2,1);
imagesc(reshape(b,sqrt(n),sqrt(n))),
colormap(gray),
title('blurred image')

%% Question 4
sigma = 0.005;

A_new = [A; sqrt(sigma)*L];
y = [b; zeros(size(L, 1), 1)];

A_mul = A_new'*A_new;

A_de = A_new'*y;
x = A_mul\A_de;

subplot(1,2,2);
imagesc(reshape(x,sqrt(n),sqrt(n))),
colormap(gray),
title('deblurred image, delta=0.005');

%%
sigma = 1e-4;

A_new = [A; sqrt(sigma)*L];
y = [b; zeros(size(L, 1), 1)];

A_mul = A_new'*A_new;

A_de = A_new'*y;
x = A_mul\A_de;

figure, subplot(1,2,1); imagesc(reshape(x,sqrt(n),sqrt(n))),
colormap(gray),
title('delta=0.0001');

%%
sigma = 0.1;

A_new = [A; sqrt(sigma)*L];
y = [b; zeros(size(L, 1), 1)];

A_mul = A_new'*A_new;

A_de = A_new'*y;
x = A_mul\A_de;

subplot(1,2,2); imagesc(reshape(x,sqrt(n),sqrt(n))),
colormap(gray),
title('delta=0.1');

