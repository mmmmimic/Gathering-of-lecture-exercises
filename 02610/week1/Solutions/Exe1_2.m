disp('The gradient is [8+2x1; 12-4x2], and the Hessian is [2, 0; 0, -4].')
disp('Set the gradient equals 0, we obtain only one stationary point [-4;3].')
disp('Since Hessian is neither positive definite nore negative definite, the stationary point is just a saddle point.')

x1=-9:0.05:1;
x2=-2:0.05:8;
[X,Y]=meshgrid(x1,x2);
F=8*X+12*Y+X.^2-2*Y.^2;

figure,
v=[-20:20];
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),