close all, clear all


%%
x0 = [-1.2;1];
gamma0 = 1;
[x,stat]=coordinate(@rosenbrock,x0,gamma0);

x_star = [1; 1];
k = 0:stat.iter;
err = sqrt((stat.X(1,:)-x_star(1)).^2+(stat.X(2,:)-x_star(2)).^2);

figure,
subplot(1,2,1), semilogy(err), title('||x_k-x^*||'),
subplot(1,2,2), semilogy(stat.F), title('f(x_k)'),



%% Contour plot
x1=-1:0.01:2;
x2=-1:0.01:2;
[X,Y]=meshgrid(x1,x2);
F=100*(Y-X.^2).^2+(1-X).^2;

figure,
v=[0:0.3:3, 5:5:100];
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),
hold on, plot(1, 1, 'r*'),
plot(stat.X(1,:),stat.X(2,:),'b.'), hold off