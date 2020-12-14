clear all; close; clc;

syms x1 x2 beta lambda 

f = 0.5*((x1-1)^2+x2^2);
c = -x1+beta*x2^2;

L = f - lambda*c;

dL = gradient(L, [x1, x2])