close all, clear all,

x0 = [1;4];  
H = eye(length(x0));
maxit = 8;
[xopt, stat] = BFGSmethod_line(H, maxit, @Myfun, x0);

k = 0:stat.iter;
err = abs(stat.X-1);

Table = [k', stat.X', stat.F']

%%

x1=-4:0.05:6;
x2=-1:0.05:9;
[X,Y]=meshgrid(x1,x2);
F=X.^4-2*X.^2.*Y+X.^2+Y.^2-2*X+5;

figure,
v=[3:0.5:20];
[c,h]=contour(X,Y,F,v,'linewidth',2);
colorbar, axis image,
xlabel('x_1','fontsize',14),
ylabel('x_2','fontsize',14),

hold on,
plot(stat.X(1,:), stat.X(2,:),'r*'),
hold off,


%%
[xSD, stat_SD] = steepestdescent_line(maxit,@Myfun,x0);
[xNT, stat_NT] = newton(1,maxit,@Myfun,x0);


Table1 = [k', stat.X', stat_SD.X', [stat_NT.X,nan(2,maxit-stat_NT.iter)]']
Table2 = [k', stat.F', stat_SD.F', [stat_NT.F,nan(1,maxit-stat_NT.iter)]']


figure,plot(0:8,stat.F,'r',0:8,stat_SD.F,'b',0:1,stat_NT.F,'g'),

%%
options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'Algorithm','trust-region',...
    'Display','iter','MaxIterations',8);
xTR = fminunc(@Myfun,x0,options);

options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'Algorithm','quasi-newton',...
    'Display','iter','MaxIterations',8);
xQN = fminunc(@Myfun,x0,options);
